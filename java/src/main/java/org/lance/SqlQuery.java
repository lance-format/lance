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

import com.google.common.base.MoreObjects;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.vector.ipc.ArrowReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class SqlQuery {
  private static final String DEFAULT_TABLE_NAME = "dataset";

  private Dataset dataset;
  private String sql;
  private String table = DEFAULT_TABLE_NAME;
  private boolean withRowId = false;
  private boolean withRowAddr = false;
  private final List<String> extraTableNames = new ArrayList<>();
  private final List<Long> extraStreamAddresses = new ArrayList<>();
  private boolean consumed = false;

  public SqlQuery(Dataset dataset, String sql) {
    this.dataset = dataset;
    this.sql = sql;
  }

  public SqlQuery tableName(String tableName) {
    this.table = tableName;
    return this;
  }

  /**
   * Register an additional in-memory Arrow relation (exported to {@code stream} via the C Data
   * Interface) as a table named {@code name}, joinable in the SQL alongside the dataset. {@link
   * #intoBatchRecords()} consumes the stream during the native call (it takes ownership of the
   * underlying C stream); the caller still owns the {@link ArrowArrayStream} handle and should
   * close it afterwards (typically via try-with-resources), as with {@code MergeInsert}. May be
   * called multiple times to register multiple tables.
   *
   * <p>The stream is fully materialized in memory before the query executes, so callers must keep
   * the input bounded to available memory. For streaming or out-of-core relations, use a custom
   * table provider through the Rust API instead of this Arrow convenience method.
   *
   * <p>{@code name} is parsed as a SQL identifier, so an unquoted name is lowercased ({@code IDs}
   * registers the table {@code ids}) while a quoted one keeps its case ({@code "IDs"} registers the
   * table {@code IDs}). If two names resolve to the same table, the later registration replaces the
   * earlier one (deduplicated when the query is built). A name that resolves to the query's own
   * {@link #tableName(String)} would hide the dataset, so {@link #intoBatchRecords()} rejects it
   * with an {@link IllegalArgumentException}.
   *
   * <p>Because the registered stream is consumed on the first {@link #intoBatchRecords()} call, a
   * query with registered relations is single-use; calling {@link #intoBatchRecords()} a second
   * time throws.
   *
   * @throws IllegalArgumentException if {@code name} is null or blank, or {@code stream} is null
   * @throws IllegalStateException if the query was already run via {@link #intoBatchRecords()}
   */
  public SqlQuery registerArrow(String name, ArrowArrayStream stream) {
    if (consumed) {
      throw new IllegalStateException(
          "registerArrow cannot be called after intoBatchRecords(); build a new query and "
              + "re-register relations.");
    }
    if (name == null || name.trim().isEmpty()) {
      throw new IllegalArgumentException(
          "registerArrow: table name must be non-empty, got: " + name);
    }
    if (stream == null) {
      throw new IllegalArgumentException(
          "registerArrow: stream must not be null (table name: " + name + ")");
    }
    this.extraTableNames.add(name);
    this.extraStreamAddresses.add(stream.memoryAddress());
    return this;
  }

  public SqlQuery withRowId(boolean withRowId) {
    this.withRowId = withRowId;
    return this;
  }

  public SqlQuery withRowAddr(boolean withAddr) {
    this.withRowAddr = withAddr;
    return this;
  }

  public ArrowReader intoBatchRecords() throws IOException {
    if (consumed) {
      throw new IllegalStateException(
          "intoBatchRecords() was already called on this SqlQuery; a query with registered "
              + "Arrow relations is single-use because the registered streams are consumed. "
              + "Build a new query and re-register them.");
    }
    try (LockManager.ReadLock readLock = dataset.acquireReadLock();
        ArrowArrayStream s = ArrowArrayStream.allocateNew(dataset.allocator())) {
      // A query with registered streams is one-shot: the native call below consumes them. Mark it
      // consumed just before that call, so a failed output-stream allocation does not brick a
      // retry.
      if (!extraTableNames.isEmpty()) {
        consumed = true;
      }
      intoBatchRecords(
          dataset,
          sql,
          Optional.ofNullable(table),
          withRowId,
          withRowAddr,
          s.memoryAddress(),
          extraTableNames,
          extraStreamAddresses);
      return Data.importArrayStream(dataset.allocator(), s);
    }
  }

  private static native void intoBatchRecords(
      Dataset dataset,
      String sql,
      Optional<String> tableName,
      boolean withRowId,
      boolean withRowAddr,
      long streamAddress,
      List<String> extraTableNames,
      List<Long> extraStreamAddresses)
      throws IOException;

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("sql", sql)
        .add("table", table)
        .add("withRowId", withRowId)
        .add("withRowAddr", withRowAddr)
        .toString();
  }
}
