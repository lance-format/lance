# MemWAL Index

The MemWAL Index is a system index that serves as the centralized structure for all MemWAL metadata.
It stores configuration (shard specs, indexes to maintain), SSTable compaction progress, and shard state snapshots.

A table has at most one MemWAL index.
The table may be a primary-key table or an append-only table without primary-key metadata.
Primary-key-dependent lookup and deduplication semantics only apply when a primary key is defined.

For the complete specification, see:

- [MemWAL Index Overview](../../table/mem_wal.md#memwal-index) - Purpose and high-level description
- [MemWAL Index Details](../../table/mem_wal.md#memwal-index-details) - Storage format, schemas, and staleness handling
- [MemWAL Implementation](../../table/mem_wal.md#implementation-expectation) - Implementation details and expectations
