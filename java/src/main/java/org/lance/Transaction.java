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
package org.lance;

import org.lance.operation.Operation;

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;

/**
 * A pure data container representing a Lance transaction.
 *
 * <p>A Transaction holds the read version, a unique identifier, the operation to perform, and
 * optional transaction properties. It does not contain commit configuration or execution logic.
 *
 * <p>To commit a transaction, use {@link CommitBuilder} or {@link SourcedTransaction}.
 */
public class Transaction implements AutoCloseable {

  private final long readVersion;
  private final String uuid;
  private final Operation operation;
  private final Optional<String> tag;
  private final Optional<Map<String, String>> transactionProperties;

  /**
   * Constructor used by JNI when reading transactions from native code.
   *
   * @param readVersion the version that was read when creating this transaction
   * @param uuid the unique identifier for this transaction
   * @param operation the operation to perform
   * @param tag optional tag for the transaction
   * @param transactionProperties optional transaction properties
   */
  private Transaction(
      long readVersion,
      String uuid,
      Operation operation,
      String tag,
      Map<String, String> transactionProperties) {
    this.readVersion = readVersion;
    this.uuid = uuid;
    this.operation = operation;
    this.tag = tag == null || tag.isEmpty() ? Optional.empty() : Optional.of(tag);
    this.transactionProperties =
        transactionProperties == null || transactionProperties.isEmpty()
            ? Optional.empty()
            : Optional.of(transactionProperties);
  }

  /**
   * Create a transaction with the given read version and operation. A random UUID is generated
   * automatically.
   *
   * @param readVersion the version that was read when creating this transaction
   * @param operation the operation to perform
   */
  public Transaction(long readVersion, Operation operation) {
    this(readVersion, UUID.randomUUID().toString(), operation, null, null);
  }

  public long readVersion() {
    return readVersion;
  }

  public String uuid() {
    return uuid;
  }

  public Operation operation() {
    return operation;
  }

  /** Returns the optional tag for this transaction; null and empty tags are both absent. */
  public Optional<String> tag() {
    return tag;
  }

  /** Returns transaction properties; null and empty maps are both absent. */
  public Optional<Map<String, String>> transactionProperties() {
    return transactionProperties;
  }

  /** Release native resources held by the operation (e.g. Arrow C schemas). */
  @Override
  public void close() {
    operation.release();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("readVersion", readVersion)
        .add("uuid", uuid)
        .add("operation", operation)
        .add("tag", tag)
        .add("transactionProperties", transactionProperties)
        .toString();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Transaction that = (Transaction) o;
    return readVersion == that.readVersion
        && uuid.equals(that.uuid)
        && Objects.equals(operation, that.operation)
        && Objects.equals(tag, that.tag)
        && Objects.equals(transactionProperties, that.transactionProperties);
  }

  @Override
  public int hashCode() {
    return Objects.hash(readVersion, uuid, operation, tag, transactionProperties);
  }

  /** Builder for constructing {@link Transaction} instances. */
  public static class Builder {
    private String uuid;
    private long readVersion;
    private Operation operation;
    private String tag;
    private Map<String, String> transactionProperties;

    public Builder() {
      this.uuid = UUID.randomUUID().toString();
    }

    public Builder readVersion(long readVersion) {
      this.readVersion = readVersion;
      return this;
    }

    public Builder uuid(String uuid) {
      this.uuid = uuid;
      return this;
    }

    public Builder operation(Operation operation) {
      if (this.operation != null) {
        throw new IllegalStateException(
            String.format("Operation %s has been set", this.operation.name()));
      }
      this.operation = operation;
      return this;
    }

    /**
     * Set an optional tag for the transaction.
     *
     * @param tag the tag string; null or empty means absent
     * @return this builder instance
     */
    public Builder tag(String tag) {
      this.tag = tag;
      return this;
    }

    /** Set transaction properties; null or an empty map means absent. */
    public Builder transactionProperties(Map<String, String> properties) {
      this.transactionProperties = properties;
      return this;
    }

    public Transaction build() {
      Preconditions.checkState(operation != null, "TransactionBuilder has no operations");
      return new Transaction(readVersion, uuid, operation, tag, transactionProperties);
    }
  }
}
