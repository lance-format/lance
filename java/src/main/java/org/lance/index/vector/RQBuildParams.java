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

/** Parameters for building a Rabit Quantizer (RQ) index stage. Defaults to 5 bits per dimension. */
public class RQBuildParams {
  private final byte numBits;

  private RQBuildParams(Builder builder) {
    this.numBits = builder.numBits;
  }

  public static class Builder {
    private byte numBits = 5;

    public Builder() {}

    /**
     * @param numBits number of bits per dimension used by Rabit quantization.
     * @return Builder
     */
    public Builder setNumBits(byte numBits) {
      this.numBits = numBits;
      return this;
    }

    public RQBuildParams build() {
      return new RQBuildParams(this);
    }
  }

  public byte getNumBits() {
    return numBits;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("numBits", numBits).toString();
  }
}
