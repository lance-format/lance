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
 * SQ8 build parameters. Defaults to 8 bits and a training sample rate of 256 (up to 256 * 256
 * vectors). A supplied model skips training and must match the vector dimension and numBits. Use
 * the same distance metric for training and building.
 */
public class SQBuildParams {
  private final short numBits;
  private final int sampleRate;
  private final Optional<SQModel> model;

  private SQBuildParams(Builder builder) {
    this.numBits = builder.numBits;
    this.sampleRate = builder.sampleRate;
    this.model = Optional.ofNullable(builder.model);
  }

  public static class Builder {
    private short numBits = 8;
    private int sampleRate = 256;
    private SQModel model;

    public Builder() {}

    /**
     * @param numBits number of bits of scaling range.
     * @return Builder
     */
    public Builder setNumBits(short numBits) {
      this.numBits = numBits;
      return this;
    }

    /**
     * @param sampleRate sample rate for training
     * @return Builder
     */
    public Builder setSampleRate(int sampleRate) {
      this.sampleRate = sampleRate;
      return this;
    }

    /** Reuse trained bounds on this worker; sampleRate is ignored when a model is supplied. */
    public Builder setModel(SQModel model) {
      this.model = Objects.requireNonNull(model, "model");
      return this;
    }

    public SQBuildParams build() {
      return new SQBuildParams(this);
    }
  }

  // Getter methods
  public short getNumBits() {
    return numBits;
  }

  public int getSampleRate() {
    return sampleRate;
  }

  /** Return the optional shared model; absent means train from the build's input. */
  public Optional<SQModel> getModel() {
    return model;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("numBits", numBits)
        .add("sampleRate", sampleRate)
        .add("model", model.isPresent())
        .toString();
  }
}
