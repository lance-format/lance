# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset SQL query builder and executable query objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, Union

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa


class SqlQuery:
    """
    An executable SQL query.

    This is created by calling :meth:`SqlQueryBuilder.build`.
    """

    def __init__(self, query):
        self._query = query

    def to_batch_records(self) -> list[pa.RecordBatch]:
        """
        Execute the query and return a list of RecordBatches.

        This is an eager operation that will load all results into memory.

        Returns
        -------
        list[pyarrow.RecordBatch]
        """
        return self._query.to_batch_records()

    def to_stream_reader(self) -> pa.RecordBatchReader:
        """
        Execute the query and return a RecordBatchReader.

        This is a lazy operation that will stream results.

        Returns
        -------
        pyarrow.RecordBatchReader
        """
        return self._query.to_stream_reader()


class SqlQueryBuilder:
    """
    A builder for SQL queries on a Lance dataset.

    This builder allows for chaining of options to configure the SQL query.
    It is returned by :meth:`LanceDataset.sql`.
    """

    def __init__(self, builder):
        self._builder = builder

    def parameters(
        self, parameters: Union[Sequence[Any], Dict[str, Any], pa.RecordBatch]
    ) -> "SqlQueryBuilder":
        """
        Bind values to SQL placeholders without interpolating SQL text.

        Parameters
        ----------
        parameters: sequence, dict, or pyarrow.RecordBatch
            A sequence binds ``$1``, ``$2``, etc. A dict binds ``$name`` using
            keys without the ``$`` prefix. Values may be Python scalars,
            including ``None``, or PyArrow scalars for explicit Arrow types.
            A RecordBatch must have exactly one row; its column names bind
            named placeholders, also without the ``$`` prefix.

            Calling this method again replaces the previous bindings. Missing
            bindings and incompatible types raise an error during execution.

        Examples
        --------
        .. code-block:: python

            query = dataset.sql(
                "SELECT * FROM dataset WHERE id = $1", parameters=[42]
            ).build()
            query.to_batch_records()

            query = dataset.sql(
                "SELECT * FROM dataset WHERE name = $name"
            ).parameters({"name": "O'Reilly"}).build()
        """
        self._builder = self._builder.parameters(parameters)
        return self

    def table_name(self, table_name: str) -> "SqlQueryBuilder":
        """
        Set the table name for the query.

        Parameters
        ----------
        table_name: str
            The name of the table to query.
        """
        self._builder = self._builder.table_name(table_name)
        return self

    def with_row_id(self, with_row_id: bool = True) -> "SqlQueryBuilder":
        """
        Include the row ID in the query result.

        Parameters
        ----------
        with_row_id: bool, default True
            Whether to include the row ID column (`_rowid`).
        """
        self._builder = self._builder.with_row_id(with_row_id)
        return self

    def with_row_addr(self, with_row_addr: bool = True) -> "SqlQueryBuilder":
        """
        Include the row address in the query result.

        Parameters
        ----------
        with_row_addr: bool, default True
            Whether to include the row address column (`_rowaddr`).
        """
        self._builder = self._builder.with_row_addr(with_row_addr)
        return self

    def blob_handling(
        self,
        blob_handling: Literal["all_binary", "blobs_descriptions", "all_descriptions"],
    ) -> "SqlQueryBuilder":
        """
        Control how blob columns are returned by this SQL query.

        - ``"all_binary"`` materializes blob columns as binary values.
        - ``"blobs_descriptions"`` returns blob descriptors (the default).
        - ``"all_descriptions"`` returns descriptions for all binary-like
          columns.
        """
        self._builder = self._builder.blob_handling(blob_handling)
        return self

    def register_arrow(
        self, name: str, data: Union[pa.Table, pa.RecordBatchReader]
    ) -> "SqlQueryBuilder":
        """
        Register an in-memory Arrow relation that the query can join.

        The relation is exposed under ``name`` and can be referenced like any
        other table in the SQL (for example joined or used in a subquery). It
        must be non-empty, since the schema is derived from the provided data.
        The input is fully materialized in memory when this method is called,
        so ``data`` must be bounded to available memory.

        ``name`` is parsed as a SQL identifier, so an unquoted name is lowercased
        (``IDs`` registers the table ``ids``) while a quoted one keeps its case
        (``"IDs"``). If two names resolve to the same table, the later
        registration wins. A name that resolves to the query's own
        :meth:`table_name` would hide the dataset and is rejected.

        Registered names are validated when the query is executed, not when
        :meth:`build` returns, so a rejected name raises from
        :meth:`SqlQuery.to_batch_records` (as ``ValueError``) or
        :meth:`SqlQuery.to_stream_reader` (as ``OSError``).

        Parameters
        ----------
        name: str
            The name the relation is registered under in the query.
        data: pyarrow.Table or pyarrow.RecordBatchReader
            The in-memory relation to register. Must be non-empty.

        Examples
        --------
        Register an in-memory relation and semi-join it in the query::

            ids = pa.table({"id": [1, 2, 3]})
            batches = (
                dataset.sql("SELECT * FROM d WHERE id IN (SELECT id FROM ids)")
                .table_name("d")
                .register_arrow("ids", ids)
                .build()
                .to_batch_records()
            )
        """
        self._builder = self._builder.register_arrow(name, data)
        return self

    def batch_size(self, batch_size: int) -> "SqlQueryBuilder":
        """
        Set the maximum number of rows produced by each query batch.

        If :meth:`batch_size_bytes` is also set, both limits apply and the one
        reached first determines the scan batch size.
        """
        self._builder = self._builder.batch_size(batch_size)
        return self

    def batch_size_bytes(self, batch_size_bytes: int) -> "SqlQueryBuilder":
        """
        Set the approximate maximum bytes produced by each scan batch.

        If :meth:`batch_size` is also set, both limits apply and the one
        reached first determines the scan batch size.
        """
        self._builder = self._builder.batch_size_bytes(batch_size_bytes)
        return self

    def build(self) -> SqlQuery:
        """
        Build the query.

        Returns
        -------
        SqlQuery
            An executable query object.
        """
        return SqlQuery(self._builder.build())
