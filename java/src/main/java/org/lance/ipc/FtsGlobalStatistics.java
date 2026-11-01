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
package org.lance.ipc;

import java.util.Arrays;
import java.util.Objects;

/**
 * Opaque protobuf-encoded global BM25 statistics for one full-text query.
 *
 * <p>The payload is bound to the dataset version, logical FTS index, exact committed segment set,
 * indexed column, document granularity, and prepared query vocabulary with Match term groups and
 * Phrase token positions. It can be transported to another Lance executor without exposing Rust
 * scorer internals as a Java API. The caller must attach it only to the same query used to produce
 * it because V1 consumers cannot independently verify the source query identity.
 *
 * <p>V1 contains one DOCUMENT-counted corpus and prepared Match/Phrase leaves. This class only
 * transports the bytes; reconstructing a handle does not validate its protobuf contents.
 *
 * <pre>{@code
 * byte[] payload = dataset.getFtsGlobalStatistics(FullTextQuery.match("hello", "text")).toBytes();
 * FtsGlobalStatistics statistics = FtsGlobalStatistics.fromBytes(payload);
 * }</pre>
 */
public final class FtsGlobalStatistics {
  private final byte[] protobuf;

  private FtsGlobalStatistics(byte[] protobuf) {
    Objects.requireNonNull(protobuf, "protobuf must not be null");
    if (protobuf.length == 0) {
      throw new IllegalArgumentException("protobuf must not be empty");
    }
    this.protobuf = Arrays.copyOf(protobuf, protobuf.length);
  }

  /**
   * Reconstruct an opaque statistics handle received from another process.
   *
   * @param protobuf nonempty protobuf payload, copied defensively
   * @return a transport handle; semantic validation belongs to the executor
   * @throws NullPointerException if protobuf is null
   * @throws IllegalArgumentException if protobuf is empty
   */
  public static FtsGlobalStatistics fromBytes(byte[] protobuf) {
    return new FtsGlobalStatistics(protobuf);
  }

  /**
   * Return a defensive copy of the protobuf transport payload.
   *
   * @return bytes that the caller may modify without changing this handle
   */
  public byte[] toBytes() {
    return Arrays.copyOf(protobuf, protobuf.length);
  }
}
