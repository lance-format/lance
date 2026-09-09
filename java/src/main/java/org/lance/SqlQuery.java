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
import java.util.Optional;

public class SqlQuery {
  private static final String DEFAULT_TABLE_NAME = "dataset";

  private Dataset dataset;
  private String sql;
  private String table = DEFAULT_TABLE_NAME;
  private boolean withRowId = false;
  private boolean withRowAddr = false;

  public SqlQuery(Dataset dataset, String sql) {
    this.dataset = dataset;
    this.sql = sql;
  }

  public SqlQuery tableName(String tableName) {
    this.table = tableName;
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
    try (LockManager.ReadLock readLock = dataset.acquireReadLock();
        ArrowArrayStream s = ArrowArrayStream.allocateNew(dataset.allocator())) {
      intoBatchRecords(
          dataset, sql, Optional.ofNullable(table), withRowId, withRowAddr, s.memoryAddress());
      return Data.importArrayStream(dataset.allocator(), s);
    }
  }

  private static native void intoBatchRecords(
      Dataset dataset,
      String sql,
      Optional<String> tableName,
      boolean withRowId,
      boolean withRowAddr,
      long streamAddress)
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
