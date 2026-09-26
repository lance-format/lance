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

import com.google.common.base.MoreObjects;

import java.util.Objects;
import java.util.Optional;

/**
 * Parameters for building a Rabit Quantizer (RQ) index stage. Defaults to 5 bits per dimension and
 * fast rotation. A supplied {@link RQModel} must match {@code numBits}, {@code rotationType}, and
 * the vector dimension of the indexed column.
 */
public class RQBuildParams {
  private final byte numBits;
  private final RQRotationType rotationType;
  private final Optional<RQModel> model;

  private RQBuildParams(Builder builder) {
    this.numBits = builder.numBits;
    this.rotationType = builder.rotationType;
    this.model = Optional.ofNullable(builder.model);
  }

  public static class Builder {
    private byte numBits = 5;
    private RQRotationType rotationType = RQRotationType.FAST;
    private RQModel model;

    public Builder() {}

    /**
     * @param numBits number of bits per dimension used by Rabit quantization.
     * @return Builder
     */
    public Builder setNumBits(byte numBits) {
      this.numBits = numBits;
      return this;
    }

    /**
     * @param rotationType rotation type used by Rabit quantization.
     * @return Builder
     */
    public Builder setRotationType(RQRotationType rotationType) {
      this.rotationType = Objects.requireNonNull(rotationType, "rotationType");
      return this;
    }

    /**
     * @param model prebuilt rotation model to reuse across builds.
     * @return Builder
     */
    public Builder setModel(RQModel model) {
      this.model = Objects.requireNonNull(model, "model");
      return this;
    }

    public RQBuildParams build() {
      return new RQBuildParams(this);
    }
  }

  public byte getNumBits() {
    return numBits;
  }

  public RQRotationType getRotationType() {
    return rotationType;
  }

  public Optional<RQModel> getModel() {
    return model;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("numBits", numBits)
        .add("rotationType", rotationType)
        .add("model", model.isPresent())
        .toString();
  }
}
