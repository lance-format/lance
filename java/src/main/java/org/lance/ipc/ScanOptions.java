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

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

/** Lance scan options. */
public class ScanOptions {
  private final Optional<List<Integer>> fragmentIds;
  private final Optional<Long> batchSize;
  private final Optional<Long> batchSizeBytes;
  private final Optional<Long> ioBufferSize;
  private final Optional<List<String>> columns;
  private final Optional<String> filter;
  private final Optional<ByteBuffer> substraitFilter;
  private final Optional<Long> limit;
  private final Optional<Long> offset;
  private final Optional<Query> nearest;
  private final Optional<FullTextQuery> fullTextQuery;
  private final boolean prefilter;
  private final boolean withRowId;
  private final boolean withRowAddress;
  private final int batchReadahead;
  private final Optional<Integer> fragmentReadahead;
  private final boolean scanInOrder;
  private final Optional<MaterializationStyle> lateMaterialization;
  private final Optional<List<ColumnOrdering>> columnOrderings;
  private final boolean useScalarIndex;
  private final Optional<ByteBuffer> substraitAggregate;
  private final boolean collectStats;
  private final boolean fastSearch;
  private final boolean includeDeletedRows;
  private final boolean strictBatchSize;
  private final boolean disableScoringAutoprojection;

  public ScanOptions(
      Optional<List<Integer>> fragmentIds,
      Optional<Long> batchSize,
      Optional<List<String>> columns,
      Optional<String> filter,
      Optional<ByteBuffer> substraitFilter,
      Optional<Long> limit,
      Optional<Long> offset,
      Optional<Query> nearest,
      Optional<FullTextQuery> fullTextQuery,
      boolean prefilter,
      boolean withRowId,
      boolean withRowAddress,
      int batchReadahead,
      Optional<List<ColumnOrdering>> columnOrderings,
      boolean useScalarIndex,
      Optional<ByteBuffer> substraitAggregate,
      boolean collectStats) {
    this(
        fragmentIds,
        batchSize,
        columns,
        filter,
        substraitFilter,
        limit,
        offset,
        nearest,
        fullTextQuery,
        prefilter,
        withRowId,
        withRowAddress,
        batchReadahead,
        columnOrderings,
        useScalarIndex,
        substraitAggregate,
        collectStats,
        false,
        false,
        false,
        false);
  }

  /**
   * Constructor for LanceScanOptions.
   *
   * @param fragmentIds the id of the fragments to scan
   * @param batchSize Maximum row number of each returned ArrowRecordBatch. Optional, use
   *     Optional.empty() if unspecified.
   * @param columns (Optional) Projected columns. Optional.empty() for scanning all columns.
   *     Otherwise, only columns present in the List will be scanned.
   * @param filter (Optional) Filter expression. Optional.empty() for no filter.
   * @param substraitFilter (Optional) Substrait filter expression.
   * @param filter (Optional) Filter expression. Optional.empty() for no filter.
   * @param limit (Optional) Maximum number of rows to return.
   * @param offset (Optional) Number of rows to skip before returning results.
   * @param withRowId Whether to include the row ID in the results.
   * @param withRowAddress Whether to include the row address in the results.
   * @param nearest (Optional) Nearest neighbor query.
   * @param batchReadahead Number of batches to read ahead.
   * @param columnOrderings (Optional) Column orderings for result sorting.
   * @param useScalarIndex Whether to use scalar indices for the scan. Default is true.
   * @param substraitAggregate (Optional) Substrait aggregate expression for aggregate pushdown.
   * @param collectStats Whether to collect scan execution statistics. Default is false.
   * @param fastSearch Whether to only search indexed fragments. Default is false.
   */
  public ScanOptions(
      Optional<List<Integer>> fragmentIds,
      Optional<Long> batchSize,
      Optional<List<String>> columns,
      Optional<String> filter,
      Optional<ByteBuffer> substraitFilter,
      Optional<Long> limit,
      Optional<Long> offset,
      Optional<Query> nearest,
      Optional<FullTextQuery> fullTextQuery,
      boolean prefilter,
      boolean withRowId,
      boolean withRowAddress,
      int batchReadahead,
      Optional<List<ColumnOrdering>> columnOrderings,
      boolean useScalarIndex,
      Optional<ByteBuffer> substraitAggregate,
      boolean collectStats,
      boolean fastSearch,
      boolean includeDeletedRows,
      boolean strictBatchSize,
      boolean disableScoringAutoprojection) {
    this(
        fragmentIds,
        batchSize,
        Optional.empty(),
        Optional.empty(),
        columns,
        filter,
        substraitFilter,
        limit,
        offset,
        nearest,
        fullTextQuery,
        prefilter,
        withRowId,
        withRowAddress,
        batchReadahead,
        Optional.empty(),
        true,
        Optional.empty(),
        columnOrderings,
        useScalarIndex,
        substraitAggregate,
        collectStats,
        fastSearch,
        includeDeletedRows,
        strictBatchSize,
        disableScoringAutoprojection);
  }

  private ScanOptions(
      Optional<List<Integer>> fragmentIds,
      Optional<Long> batchSize,
      Optional<Long> batchSizeBytes,
      Optional<Long> ioBufferSize,
      Optional<List<String>> columns,
      Optional<String> filter,
      Optional<ByteBuffer> substraitFilter,
      Optional<Long> limit,
      Optional<Long> offset,
      Optional<Query> nearest,
      Optional<FullTextQuery> fullTextQuery,
      boolean prefilter,
      boolean withRowId,
      boolean withRowAddress,
      int batchReadahead,
      Optional<Integer> fragmentReadahead,
      boolean scanInOrder,
      Optional<MaterializationStyle> lateMaterialization,
      Optional<List<ColumnOrdering>> columnOrderings,
      boolean useScalarIndex,
      Optional<ByteBuffer> substraitAggregate,
      boolean collectStats,
      boolean fastSearch,
      boolean includeDeletedRows,
      boolean strictBatchSize,
      boolean disableScoringAutoprojection) {
    Preconditions.checkArgument(
        !(filter.isPresent() && substraitFilter.isPresent()),
        "cannot set both substrait filter and string filter");
    Preconditions.checkArgument(
        batchReadahead > 0, "batchReadahead must be greater than 0, got %s", batchReadahead);
    batchSizeBytes.ifPresent(
        value ->
            Preconditions.checkArgument(
                value > 0, "batchSizeBytes must be greater than 0, got %s", value));
    ioBufferSize.ifPresent(
        value ->
            Preconditions.checkArgument(
                value > 0, "ioBufferSize must be greater than 0, got %s", value));
    fragmentReadahead.ifPresent(
        value ->
            Preconditions.checkArgument(
                value > 0, "fragmentReadahead must be greater than 0, got %s", value));
    Preconditions.checkArgument(
        !(strictBatchSize && batchSizeBytes.isPresent()),
        "strictBatchSize=true cannot be combined with batchSizeBytes");
    this.fragmentIds = fragmentIds;
    this.batchSize = batchSize;
    this.batchSizeBytes = batchSizeBytes;
    this.ioBufferSize = ioBufferSize;
    this.columns = columns;
    this.filter = filter;
    this.substraitFilter = substraitFilter;
    this.limit = limit;
    this.offset = offset;
    this.nearest = nearest;
    this.fullTextQuery = fullTextQuery;
    this.prefilter = prefilter;
    this.withRowId = withRowId;
    this.withRowAddress = withRowAddress;
    this.batchReadahead = batchReadahead;
    this.fragmentReadahead = fragmentReadahead;
    this.scanInOrder = scanInOrder;
    this.lateMaterialization = lateMaterialization;
    this.columnOrderings = columnOrderings;
    this.useScalarIndex = useScalarIndex;
    this.substraitAggregate = substraitAggregate;
    this.collectStats = collectStats;
    this.fastSearch = fastSearch;
    this.includeDeletedRows = includeDeletedRows;
    this.strictBatchSize = strictBatchSize;
    this.disableScoringAutoprojection = disableScoringAutoprojection;
  }

  /**
   * Get the fragment ids.
   *
   * @return Optional containing the fragment ids if specified, otherwise empty.
   */
  public Optional<List<Integer>> getFragmentIds() {
    return fragmentIds;
  }

  /**
   * Get the batch size.
   *
   * @return Optional containing the batch size if specified, otherwise empty.
   */
  public Optional<Long> getBatchSize() {
    return batchSize;
  }

  /**
   * Get the target batch size in bytes.
   *
   * @return Optional containing the target batch size in bytes if specified, otherwise empty.
   */
  public Optional<Long> getBatchSizeBytes() {
    return batchSizeBytes;
  }

  /**
   * Get the I/O buffer size in bytes.
   *
   * @return Optional containing the I/O buffer size if specified, otherwise empty.
   */
  public Optional<Long> getIoBufferSize() {
    return ioBufferSize;
  }

  /**
   * Get the columns.
   *
   * @return Optional containing the columns if specified, otherwise empty.
   */
  public Optional<List<String>> getColumns() {
    return columns;
  }

  /**
   * Get the filter.
   *
   * @return Optional containing the filter if specified, otherwise empty.
   */
  public Optional<String> getFilter() {
    return filter;
  }

  /**
   * Get the substrait filter.
   *
   * @return Optional containing the substrait filter if specified, otherwise empty.
   */
  public Optional<ByteBuffer> getSubstraitFilter() {
    return substraitFilter;
  }

  /**
   * Get the limit.
   *
   * @return Optional containing the limit if specified, otherwise empty.
   */
  public Optional<Long> getLimit() {
    return limit;
  }

  /**
   * Get the offset.
   *
   * @return Optional containing the offset if specified, otherwise empty.
   */
  public Optional<Long> getOffset() {
    return offset;
  }

  /**
   * Get the nearest neighbor query.
   *
   * @return Optional containing the nearest neighbor query if specified, otherwise empty.
   */
  public Optional<Query> getNearest() {
    return nearest;
  }

  /**
   * Get the full text search query.
   *
   * @return Optional containing the full text search query if specified, otherwise empty.
   */
  public Optional<FullTextQuery> getFullTextQuery() {
    return fullTextQuery;
  }

  /**
   * Get whether to prefilter before nearest neighbor search.
   *
   * @return true if prefilter should be applied, false otherwise.
   */
  public boolean isPrefilter() {
    return prefilter;
  }

  /**
   * Get whether to include the row ID.
   *
   * @return true if row ID should be included, false otherwise.
   */
  public boolean isWithRowId() {
    return withRowId;
  }

  /**
   * Get whether to include the row address.
   *
   * @return true if row address should be included, false otherwise.
   */
  public boolean isWithRowAddress() {
    return withRowAddress;
  }

  /**
   * Get the batch readahead.
   *
   * @return the number of batches to read ahead.
   */
  public int getBatchReadahead() {
    return batchReadahead;
  }

  /**
   * Get the number of fragments to read ahead concurrently.
   *
   * @return Optional containing the fragment readahead if specified, otherwise empty.
   */
  public Optional<Integer> getFragmentReadahead() {
    return fragmentReadahead;
  }

  /**
   * Get whether batches should be returned in fragment and batch order.
   *
   * @return true to preserve scan order, false to return batches as soon as they are ready.
   */
  public boolean isScanInOrder() {
    return scanInOrder;
  }

  /**
   * Get the materialization policy for projected columns.
   *
   * @return Optional containing the materialization policy if specified, otherwise empty.
   */
  public Optional<MaterializationStyle> getLateMaterialization() {
    return lateMaterialization;
  }

  public Optional<List<ColumnOrdering>> getColumnOrderings() {
    return columnOrderings;
  }

  /**
   * Get whether to use scalar indices for the scan.
   *
   * @return true if scalar indices should be used, false otherwise.
   */
  public boolean isUseScalarIndex() {
    return useScalarIndex;
  }

  /**
   * Get whether to only search indexed fragments.
   *
   * @return true if unindexed fragments should be skipped, false otherwise.
   */
  public boolean isFastSearch() {
    return fastSearch;
  }

  /**
   * Get the substrait aggregate expression.
   *
   * @return Optional containing the substrait aggregate if specified, otherwise empty.
   */
  public Optional<ByteBuffer> getSubstraitAggregate() {
    return substraitAggregate;
  }

  public boolean isCollectStats() {
    return collectStats;
  }

  /**
   * Get whether to include deleted rows in scan results.
   *
   * @return true if deleted rows should be included, false otherwise.
   */
  public boolean isIncludeDeletedRows() {
    return includeDeletedRows;
  }

  /**
   * Get whether to enforce strict batch sizing.
   *
   * @return true if batch sizes must be strictly enforced, false otherwise.
   */
  public boolean isStrictBatchSize() {
    return strictBatchSize;
  }

  /**
   * Get whether to disable scoring autoprojection.
   *
   * @return true if scoring column autoprojection is disabled, false otherwise.
   */
  public boolean isDisableScoringAutoprojection() {
    return disableScoringAutoprojection;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragmentIds", fragmentIds.orElse(null))
        .add("batchSize", batchSize.orElse(null))
        .add("batchSizeBytes", batchSizeBytes.orElse(null))
        .add("ioBufferSize", ioBufferSize.orElse(null))
        .add("columns", columns.orElse(null))
        .add("filter", filter.orElse(null))
        .add(
            "substraitFilter",
            substraitFilter.map(buf -> "ByteBuffer[" + buf.remaining() + " bytes]").orElse(null))
        .add("limit", limit.orElse(null))
        .add("offset", offset.orElse(null))
        .add("nearest", nearest.orElse(null))
        .add("fullTextQuery", fullTextQuery.orElse(null))
        .add("prefilter", prefilter)
        .add("withRowId", withRowId)
        .add("WithRowAddress", withRowAddress)
        .add("batchReadahead", batchReadahead)
        .add("fragmentReadahead", fragmentReadahead.orElse(null))
        .add("scanInOrder", scanInOrder)
        .add("lateMaterialization", lateMaterialization.orElse(null))
        .add("columnOrdering", columnOrderings)
        .add("useScalarIndex", useScalarIndex)
        .add("fastSearch", fastSearch)
        .add(
            "substraitAggregate",
            substraitAggregate.map(buf -> "ByteBuffer[" + buf.remaining() + " bytes]").orElse(null))
        .add("collectStats", collectStats)
        .add("includeDeletedRows", includeDeletedRows)
        .add("strictBatchSize", strictBatchSize)
        .add("disableScoringAutoprojection", disableScoringAutoprojection)
        .toString();
  }

  /** Builder for constructing LanceScanOptions. */
  public static class Builder {
    private Optional<List<Integer>> fragmentIds = Optional.empty();
    private Optional<Long> batchSize = Optional.empty();
    private Optional<Long> batchSizeBytes = Optional.empty();
    private Optional<Long> ioBufferSize = Optional.empty();
    private Optional<List<String>> columns = Optional.empty();
    private Optional<String> filter = Optional.empty();
    private Optional<ByteBuffer> substraitFilter = Optional.empty();
    private Optional<Long> limit = Optional.empty();
    private Optional<Long> offset = Optional.empty();
    private Optional<Query> nearest = Optional.empty();
    private Optional<FullTextQuery> fullTextQuery = Optional.empty();
    private boolean prefilter = false;
    private boolean withRowId = false;
    private boolean withRowAddress = false;
    private int batchReadahead = 16;
    private Optional<Integer> fragmentReadahead = Optional.empty();
    private boolean scanInOrder = true;
    private Optional<MaterializationStyle> lateMaterialization = Optional.empty();
    private Optional<List<ColumnOrdering>> columnOrderings = Optional.empty();
    private boolean useScalarIndex = true;
    private boolean fastSearch = false;
    private Optional<ByteBuffer> substraitAggregate = Optional.empty();
    private boolean collectStats = false;
    private boolean includeDeletedRows = false;
    private boolean strictBatchSize = false;
    private boolean disableScoringAutoprojection = false;

    public Builder() {}

    /**
     * Create a builder from another scan options.
     *
     * @param options another scan options
     */
    public Builder(ScanOptions options) {
      this.fragmentIds = options.getFragmentIds();
      this.batchSize = options.getBatchSize();
      this.batchSizeBytes = options.getBatchSizeBytes();
      this.ioBufferSize = options.getIoBufferSize();
      this.columns = options.getColumns();
      this.filter = options.getFilter();
      this.substraitFilter = options.getSubstraitFilter();
      this.limit = options.getLimit();
      this.offset = options.getOffset();
      this.nearest = options.getNearest();
      this.fullTextQuery = options.getFullTextQuery();
      this.prefilter = options.isPrefilter();
      this.withRowId = options.isWithRowId();
      this.withRowAddress = options.isWithRowAddress();
      this.batchReadahead = options.getBatchReadahead();
      this.fragmentReadahead = options.getFragmentReadahead();
      this.scanInOrder = options.isScanInOrder();
      this.lateMaterialization = options.getLateMaterialization();
      this.columnOrderings = options.getColumnOrderings();
      this.useScalarIndex = options.isUseScalarIndex();
      this.fastSearch = options.isFastSearch();
      this.substraitAggregate = options.getSubstraitAggregate();
      this.collectStats = options.isCollectStats();
      this.includeDeletedRows = options.isIncludeDeletedRows();
      this.strictBatchSize = options.isStrictBatchSize();
      this.disableScoringAutoprojection = options.isDisableScoringAutoprojection();
    }

    /**
     * Set the fragment ids.
     *
     * @param fragmentIds the id of the fragments to scan
     * @return Builder instance for method chaining.
     */
    public Builder fragmentIds(List<Integer> fragmentIds) {
      this.fragmentIds = Optional.of(fragmentIds);
      return this;
    }

    /**
     * Set the batch size.
     *
     * @param batchSize Maximum row number of each returned ArrowRecordBatch.
     * @return Builder instance for method chaining.
     */
    public Builder batchSize(long batchSize) {
      this.batchSize = Optional.of(batchSize);
      return this;
    }

    /**
     * Set the approximate target size of each returned Arrow record batch in bytes.
     *
     * <p>If {@link #batchSize(long)} is also set, the limit reached first determines the batch
     * size. This option cannot be combined with {@link #strictBatchSize(boolean)}.
     *
     * @param batchSizeBytes target batch size in bytes; must be greater than zero.
     * @return Builder instance for method chaining.
     */
    public Builder batchSizeBytes(long batchSizeBytes) {
      this.batchSizeBytes = Optional.of(batchSizeBytes);
      return this;
    }

    /**
     * Set the amount of memory reserved for buffering storage I/O.
     *
     * <p>This controls scanner backpressure but is not a hard upper bound on total scanner memory.
     * It currently applies only to v2 files.
     *
     * @param ioBufferSize I/O buffer size in bytes; must be greater than zero.
     * @return Builder instance for method chaining.
     */
    public Builder ioBufferSize(long ioBufferSize) {
      this.ioBufferSize = Optional.of(ioBufferSize);
      return this;
    }

    /**
     * Set the columns.
     *
     * @param columns Projected columns.
     * @return Builder instance for method chaining.
     */
    public Builder columns(List<String> columns) {
      this.columns = Optional.of(columns);
      return this;
    }

    /**
     * Set the filter.
     *
     * @param filter Filter expression.
     * @return Builder instance for method chaining.
     */
    public Builder filter(String filter) {
      this.filter = Optional.of(filter);
      return this;
    }

    /**
     * Set the substrait filter.
     *
     * @param substraitFilter Substrait filter expression.
     * @return Builder instance for method chaining.
     */
    public Builder substraitFilter(ByteBuffer substraitFilter) {
      this.substraitFilter = Optional.of(substraitFilter);
      return this;
    }

    /**
     * Set the limit.
     *
     * @param limit Maximum number of rows to return.
     * @return Builder instance for method chaining.
     */
    public Builder limit(long limit) {
      this.limit = Optional.of(limit);
      return this;
    }

    /**
     * Set the offset.
     *
     * @param offset Number of rows to skip before returning results.
     * @return Builder instance for method chaining.
     */
    public Builder offset(long offset) {
      this.offset = Optional.of(offset);
      return this;
    }

    /**
     * Set the nearest neighbor query.
     *
     * @param nearest The nearest neighbor query.
     * @return Builder instance for method chaining.
     */
    public Builder nearest(Query nearest) {
      this.nearest = Optional.of(nearest);
      return this;
    }

    /**
     * Set the full text search query.
     *
     * @param fullTextQuery full text search query definition.
     * @return Builder instance for method chaining.
     */
    public Builder fullTextQuery(FullTextQuery fullTextQuery) {
      this.fullTextQuery = Optional.ofNullable(fullTextQuery);
      return this;
    }

    /**
     * Set whether to prefilter during nearest neighbor search.
     *
     * @param prefilter true to apply prefilter, false otherwise.
     * @return Builder instance for method chaining.
     */
    public Builder prefilter(boolean prefilter) {
      this.prefilter = prefilter;
      return this;
    }

    /**
     * Set whether to include the row ID.
     *
     * @param withRowId true to include row ID, false otherwise.
     * @return Builder instance for method chaining.
     */
    public Builder withRowId(boolean withRowId) {
      this.withRowId = withRowId;
      return this;
    }

    /**
     * Set whether to include the row addr.
     *
     * @param withRowAddress true to include row ID, false otherwise.
     * @return Builder instance for method chaining.
     */
    public Builder withRowAddress(boolean withRowAddress) {
      this.withRowAddress = withRowAddress;
      return this;
    }

    /**
     * Set the batch readahead.
     *
     * @param batchReadahead Number of batches to read ahead.
     * @return Builder instance for method chaining.
     */
    public Builder batchReadahead(int batchReadahead) {
      this.batchReadahead = batchReadahead;
      return this;
    }

    /**
     * Set the number of fragments to read ahead concurrently.
     *
     * @param fragmentReadahead number of fragments to read ahead; must be greater than zero.
     * @return Builder instance for method chaining.
     */
    public Builder fragmentReadahead(int fragmentReadahead) {
      this.fragmentReadahead = Optional.of(fragmentReadahead);
      return this;
    }

    /**
     * Set whether batches are returned in fragment and batch order.
     *
     * <p>Disabling ordering can increase scan concurrency at the cost of additional memory. This
     * option is ignored for v2 files, which always scan in order. It is also ignored when a column
     * ordering or nearest-neighbor query is configured; the query determines the result order.
     *
     * @param scanInOrder true to preserve scan order, false to return ready batches immediately.
     * @return Builder instance for method chaining.
     */
    public Builder scanInOrder(boolean scanInOrder) {
      this.scanInOrder = scanInOrder;
      return this;
    }

    /**
     * Set when projected columns that are not used by the filter should be materialized.
     *
     * <p>This option only affects regular scans. Vector search and full-text search always use late
     * materialization.
     *
     * @param lateMaterialization materialization policy.
     * @return Builder instance for method chaining.
     */
    public Builder lateMaterialization(MaterializationStyle lateMaterialization) {
      this.lateMaterialization = Optional.of(lateMaterialization);
      return this;
    }

    public Builder setColumnOrderings(List<ColumnOrdering> columnOrderings) {
      this.columnOrderings = Optional.of(columnOrderings);
      return this;
    }

    /**
     * Set whether to use scalar indices for the scan.
     *
     * <p>Scans will use scalar indices, when available, to optimize queries with filters. However,
     * in some corner cases, scalar indices may make performance worse. This parameter allows users
     * to disable scalar indices in these cases.
     *
     * @param useScalarIndex true to use scalar indices, false otherwise. Default is true.
     * @return Builder instance for method chaining.
     */
    public Builder useScalarIndex(boolean useScalarIndex) {
      this.useScalarIndex = useScalarIndex;
      return this;
    }

    /**
     * Set whether to only search indexed fragments.
     *
     * <p>This is a weak-consistency mode for vector search, full text search, and scalar-indexed
     * filters. It can reduce latency by skipping recently appended fragments that are not covered
     * by the relevant index.
     *
     * @param fastSearch true to skip unindexed fragments, false otherwise. Default is false.
     * @return Builder instance for method chaining.
     */
    public Builder fastSearch(boolean fastSearch) {
      this.fastSearch = fastSearch;
      return this;
    }

    /**
     * Set the substrait aggregate expression.
     *
     * @param substraitAggregate Substrait aggregate expression.
     * @return Builder instance for method chaining.
     */
    public Builder substraitAggregate(ByteBuffer substraitAggregate) {
      this.substraitAggregate = Optional.of(substraitAggregate);
      return this;
    }

    /**
     * Enable or disable scan execution statistics collection.
     *
     * <p>When enabled, the native scanner will collect statistics (see {@link ScanStats}) for the
     * scan and make them available via {@link LanceScanner#getStats()} after the scan stream is
     * fully consumed and closed.
     *
     * <p>Default is false.
     */
    public Builder collectStats(boolean collectStats) {
      this.collectStats = collectStats;
      return this;
    }

    /**
     * Set whether to include deleted rows in scan results. Default is false.
     *
     * @param includeDeletedRows whether to include deleted rows
     * @return Builder instance for method chaining.
     */
    public Builder includeDeletedRows(boolean includeDeletedRows) {
      this.includeDeletedRows = includeDeletedRows;
      return this;
    }

    /**
     * Set whether to enforce strict batch sizing. Default is false.
     *
     * @param strictBatchSize whether to enforce strict batch sizing
     * @return Builder instance for method chaining.
     */
    public Builder strictBatchSize(boolean strictBatchSize) {
      this.strictBatchSize = strictBatchSize;
      return this;
    }

    /**
     * Set whether to disable scoring column autoprojection. Default is false.
     *
     * @param disableScoringAutoprojection whether to disable autoprojection
     * @return Builder instance for method chaining.
     */
    public Builder disableScoringAutoprojection(boolean disableScoringAutoprojection) {
      this.disableScoringAutoprojection = disableScoringAutoprojection;
      return this;
    }

    /**
     * Build the LanceScanOptions instance.
     *
     * @return LanceScanOptions instance with the specified parameters.
     */
    public ScanOptions build() {
      return new ScanOptions(
          fragmentIds,
          batchSize,
          batchSizeBytes,
          ioBufferSize,
          columns,
          filter,
          substraitFilter,
          limit,
          offset,
          nearest,
          fullTextQuery,
          prefilter,
          withRowId,
          withRowAddress,
          batchReadahead,
          fragmentReadahead,
          scanInOrder,
          lateMaterialization,
          columnOrderings,
          useScalarIndex,
          substraitAggregate,
          collectStats,
          fastSearch,
          includeDeletedRows,
          strictBatchSize,
          disableScoringAutoprojection);
    }
  }
}
