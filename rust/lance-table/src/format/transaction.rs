// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Transaction struct for lance-table format layer.
//!
//! This struct is introduced to provide a Struct-first API for passing transaction
//! information within the lance-table crate. It mirrors the protobuf Transaction
//! message at a semantic level while remaining crate-local, so lance-table does
//! not depend on higher layers (e.g., lance crate).
//!
//! Conversion to protobuf occurs at the write boundary. See the `From<Transaction>`
//! implementation below.

use crate::format::pb;

#[derive(Clone, Debug, PartialEq)]
pub struct Transaction {
    /// Crate-local representation backing: protobuf Transaction.
    /// Keeping this simple avoids ring dependencies while still enabling
    /// Struct-first parameter passing in lance-table.
    pub inner: pb::Transaction,
}

impl Transaction {
    /// Accessor for testing or internal inspection if needed.
    pub fn as_pb(&self) -> &pb::Transaction {
        &self.inner
    }

    /// Whether this transaction can change the schema, and so can introduce or
    /// worsen an invalid primary key.
    ///
    /// The rest leave the key exactly as they found it, so a table that already
    /// carries an invalid one stays writable through them -- including the
    /// deletes needed to repair it. An unrecognized operation counts as
    /// schema-changing: an unknown write is not a safe one to exempt.
    pub fn may_change_schema(&self) -> bool {
        operation_may_change_schema(&self.inner)
    }
}

/// The same classification for a protobuf that has not been wrapped yet.
///
/// The commit path has to classify the operation before it knows whether the
/// encoded bytes are small enough to inline into the manifest. Reading the
/// disposition off the inline copy instead would tie it to the payload size,
/// so the identical operation would be classified one way under the inline
/// limit and the other way above it.
pub fn operation_may_change_schema(transaction: &pb::Transaction) -> bool {
    use pb::transaction::Operation;
    !matches!(
        transaction.operation.as_ref(),
        Some(
            Operation::Append(_)
                | Operation::Delete(_)
                | Operation::CreateIndex(_)
                | Operation::Rewrite(_)
                | Operation::DataReplacement(_)
                | Operation::ReserveFragments(_)
                | Operation::Update(_)
                | Operation::UpdateConfig(_)
                | Operation::UpdateMemWalState(_)
                | Operation::UpdateBases(_)
                | Operation::DataOverlay(_)
        )
    )
}

/// Write-boundary conversion: serialize using protobuf at the last step.
impl From<Transaction> for pb::Transaction {
    fn from(tx: Transaction) -> Self {
        tx.inner
    }
}

impl From<pb::Transaction> for Transaction {
    fn from(pb_tx: pb::Transaction) -> Self {
        Self { inner: pb_tx }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    /// The classification that must never depend on payload size. A MemWAL
    /// table's transactions carry mem-table state and routinely outgrow the
    /// inline limit, so an exempt operation has to stay exempt while large --
    /// otherwise the deletes that repair an invalid key are blocked on exactly
    /// the tables most likely to have one.
    #[test]
    fn an_exempt_operation_is_classified_the_same_at_any_size() {
        let small = pb::Transaction {
            operation: Some(pb::transaction::Operation::Delete(
                pb::transaction::Delete::default(),
            )),
            ..Default::default()
        };
        let mut large = small.clone();
        large.tag = "x".repeat(4 * 1024 * 1024);

        assert!(large.encoded_len() > small.encoded_len() * 100);
        assert!(!operation_may_change_schema(&small));
        assert!(!operation_may_change_schema(&large));
    }

    /// An overlay attaches files to existing fragments and supplies new cell
    /// values; it carries no schema. Omitting it left a legacy nullable-key
    /// dataset unable to commit one, which is the upgrade path this exemption
    /// exists to keep open.
    #[test]
    fn a_data_overlay_is_exempt() {
        let overlay = pb::Transaction {
            operation: Some(pb::transaction::Operation::DataOverlay(
                pb::transaction::DataOverlay::default(),
            )),
            ..Default::default()
        };
        assert!(!operation_may_change_schema(&overlay));
    }

    /// And the converse, so the exemption cannot silently widen to everything.
    #[test]
    fn a_schema_carrying_operation_is_never_exempt() {
        let overwrite = pb::Transaction {
            operation: Some(pb::transaction::Operation::Overwrite(
                pb::transaction::Overwrite::default(),
            )),
            ..Default::default()
        };
        assert!(operation_may_change_schema(&overwrite));

        // An operation this build does not recognise must not be exempt
        // either: an unknown write is not a safe one to skip.
        let unknown = pb::Transaction::default();
        assert!(operation_may_change_schema(&unknown));
    }
}
