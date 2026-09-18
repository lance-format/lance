// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The transaction itself: an operation plus the version it was based on.

use crate::transaction::Operation;
use lance_core::deepsize::DeepSizeOf;
use lance_select::RowAddrTreeMap;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// A change to a dataset that can be retried
///
/// This contains enough information to be able to build the next manifest,
/// given the current manifest.
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct Transaction {
    /// The version of the table this transaction is based off of. If this is
    /// the first transaction, this should be 0.
    pub read_version: u64,
    pub uuid: String,
    pub operation: Operation,
    pub tag: Option<String>,
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,
    /// Conditions that must hold for this transaction to commit. A commit is
    /// rejected instead of rebased when a concurrent transaction violates one.
    pub preconditions: Vec<Precondition>,
}

/// A declaration that specific cells of the dataset must be unchanged when
/// this transaction commits.
///
/// `rows` holds the protected row addresses; `field_ids` the protected
/// columns. If a concurrent transaction modified any protected cell, this
/// transaction's commit is rejected rather than rebased, since a staged value
/// computed from the old state would be stale.
#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct Precondition {
    pub field_ids: Vec<i32>,
    pub rows: RowAddrTreeMap,
}

/// Add TransactionBuilder for flexibly setting option without using `mut`
pub struct TransactionBuilder {
    read_version: u64,
    // uuid is optional for builder since it can autogenerate
    uuid: Option<String>,
    operation: Operation,
    tag: Option<String>,
    transaction_properties: Option<Arc<HashMap<String, String>>>,
    preconditions: Vec<Precondition>,
}

impl TransactionBuilder {
    pub fn new(read_version: u64, operation: Operation) -> Self {
        Self {
            read_version,
            uuid: None,
            operation,
            tag: None,
            transaction_properties: None,
            preconditions: Vec::new(),
        }
    }

    pub fn precondition(mut self, precondition: Precondition) -> Self {
        self.preconditions.push(precondition);
        self
    }

    pub fn uuid(mut self, uuid: String) -> Self {
        self.uuid = Some(uuid);
        self
    }

    pub fn tag(mut self, tag: Option<String>) -> Self {
        self.tag = tag;
        self
    }

    pub fn transaction_properties(
        mut self,
        transaction_properties: Option<Arc<HashMap<String, String>>>,
    ) -> Self {
        self.transaction_properties = transaction_properties;
        self
    }

    pub fn build(self) -> Transaction {
        let uuid = self
            .uuid
            .unwrap_or_else(|| Uuid::new_v4().hyphenated().to_string());
        Transaction {
            read_version: self.read_version,
            uuid,
            operation: self.operation,
            tag: self.tag,
            transaction_properties: self.transaction_properties,
            preconditions: self.preconditions,
        }
    }
}

impl Transaction {
    pub fn new_from_version(read_version: u64, operation: Operation) -> Self {
        TransactionBuilder::new(read_version, operation).build()
    }

    pub fn new(read_version: u64, operation: Operation, tag: Option<String>) -> Self {
        TransactionBuilder::new(read_version, operation)
            .tag(tag)
            .build()
    }
}
