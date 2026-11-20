// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Admit generated lineage values before reading the corresponding payload.

use std::{fmt::Debug, ops::Range, sync::Arc};

use arrow_array::ArrayRef;
use arrow_schema::FieldRef;
use lance_core::{
    Error, ROW_CREATED_AT_VERSION_FIELD, ROW_LAST_UPDATED_AT_VERSION_FIELD, Result,
    utils::allocation::Reservation,
};

use super::{FullZipScan, row_ids::row_array};

/// Immutable dataset versions indexed by physical file-row position.
///
/// Implementations stream exactly the requested values using bounded iterator
/// state over shared metadata, without collecting output or blocking for I/O or
/// allocation credit. Length and mapping must stay constant for this source's
/// lifetime. This is distinct from stable row IDs: version values may repeat.
///
/// ```
/// use std::sync::Arc;
/// use lance_core::Result;
/// use lance_file::reader::{FullZipScan, PhysicalRowVersionSource};
/// fn attach(scan: FullZipScan, versions: Arc<dyn PhysicalRowVersionSource>) -> Result<FullZipScan> {
///     scan.with_row_created_at_version(Some(versions))
/// }
/// ```
pub trait PhysicalRowVersionSource: Debug + Send + Sync {
    /// Physical rows covered, including positions subsequently deleted.
    fn num_rows(&self) -> u64;

    /// Return one version per physical position, rejecting invalid/reversed bounds.
    /// The file reader verifies cardinality before delivering a complete batch.
    fn select_range(&self, rows: Range<usize>) -> Result<Box<dyn Iterator<Item = u64> + '_>>;
}

#[derive(Debug, Clone)]
enum VersionSource {
    Default,
    Sequence(Arc<dyn PhysicalRowVersionSource>),
}

impl VersionSource {
    fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Default, Self::Default) => true,
            (Self::Sequence(left), Self::Sequence(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    fn array(&self, rows: Range<usize>, credit: &mut Reservation) -> Result<ArrayRef> {
        match self {
            Self::Default => {
                row_array(credit, rows.len(), std::iter::repeat_n(1, rows.len()), None)
            }
            Self::Sequence(source) => {
                row_array(credit, rows.len(), source.select_range(rows)?, None)
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct RowVersionColumns {
    updated: Option<VersionSource>,
    created: Option<VersionSource>,
}

pub(super) struct PreparedRowVersions {
    columns: RowVersionColumns,
    rows: Range<usize>,
    credit: Reservation,
}

enum VersionKind {
    Updated,
    Created,
}

impl FullZipScan {
    /// Append `_row_created_at_version` within the batch's allocation workspace.
    ///
    /// `None` produces version 1, matching the default fragment lineage semantics.
    /// A supplied snapshot covers all physical rows, including deleted positions.
    /// The caller must bind the correct fragment's metadata; length equality is
    /// not identity. Configure after merging files and before any read attempt or
    /// logical selection. Supplied metadata stays outside the payload budget.
    ///
    /// Version outputs keep their allocation charge through final Arrow aliases.
    /// Created/updated columns share one buffer when both use the default or the
    /// exact same source object. This adds no ordinary scanner/fragment adoption.
    ///
    /// ```
    /// use lance_core::Result;
    /// use lance_file::reader::FullZipScan;
    /// fn attach(scan: FullZipScan) -> Result<FullZipScan> {
    ///     scan.with_row_created_at_version(None)
    /// }
    /// ```
    pub fn with_row_created_at_version(
        self,
        source: Option<Arc<dyn PhysicalRowVersionSource>>,
    ) -> Result<Self> {
        self.with_row_version(VersionKind::Created, source)
    }

    /// Append `_row_last_updated_at_version`; the same snapshot, default and
    /// lifetime rules as [`Self::with_row_created_at_version`] apply.
    /// Generated fields always follow payload, row ID, row address, updated
    /// version, created version order, regardless of configuration order.
    ///
    /// ```
    /// use lance_core::Result;
    /// use lance_file::reader::FullZipScan;
    /// fn attach(scan: FullZipScan) -> Result<FullZipScan> {
    ///     scan.with_row_last_updated_at_version(None)
    /// }
    /// ```
    pub fn with_row_last_updated_at_version(
        self,
        source: Option<Arc<dyn PhysicalRowVersionSource>>,
    ) -> Result<Self> {
        self.with_row_version(VersionKind::Updated, source)
    }

    fn with_row_version(
        mut self,
        kind: VersionKind,
        source: Option<Arc<dyn PhysicalRowVersionSource>>,
    ) -> Result<Self> {
        if self.has_started || usize::try_from(self.read_len).is_err() {
            return Err(Error::invalid_input(
                "FullZip versions require addressable rows and configuration before the first read or logical selection",
            ));
        }
        if let Some(source) = &source
            && source.num_rows() != self.read_len
        {
            return Err(Error::invalid_input(
                "FullZip version sequence length differs from physical length",
            ));
        }
        let columns = self.row_versions.get_or_insert_default();
        let target = match kind {
            VersionKind::Updated => &mut columns.updated,
            VersionKind::Created => &mut columns.created,
        };
        if target.is_some() {
            return Err(Error::invalid_input(
                "FullZip version column is already configured",
            ));
        }
        *target = Some(
            source
                .map(VersionSource::Sequence)
                .unwrap_or(VersionSource::Default),
        );
        self.update_generated_schema()?;
        Ok(self)
    }
}

impl RowVersionColumns {
    pub(super) fn append_fields(&self, fields: &mut Vec<FieldRef>) -> Result<()> {
        for (source, field) in [
            (&self.updated, &*ROW_LAST_UPDATED_AT_VERSION_FIELD),
            (&self.created, &*ROW_CREATED_AT_VERSION_FIELD),
        ] {
            if source.is_some() {
                if fields
                    .iter()
                    .any(|existing| existing.name() == field.name())
                {
                    return Err(Error::invalid_input(format!(
                        "FullZip generated field conflicts with projected field {}",
                        field.name()
                    )));
                }
                fields.push(Arc::new(field.clone()));
            }
        }
        Ok(())
    }

    fn shared_source(&self) -> bool {
        self.updated
            .as_ref()
            .zip(self.created.as_ref())
            .is_some_and(|(left, right)| left.same_as(right))
    }

    pub(super) fn prepare(
        &self,
        rows: Range<u64>,
        workspace: &mut Reservation,
    ) -> Result<Option<PreparedRowVersions>> {
        let rows = rows.start as usize..rows.end as usize;
        let buffers = usize::from(self.updated.is_some()) + usize::from(self.created.is_some())
            - usize::from(self.shared_source());
        let bytes = rows
            .len()
            .checked_mul(8)
            .and_then(|bytes| bytes.checked_mul(buffers))
            .ok_or_else(|| Error::invalid_input("FullZip version column allocation overflows"))?;
        Ok(workspace
            .try_split(bytes)
            .map(|credit| PreparedRowVersions {
                columns: self.clone(),
                rows,
                credit,
            }))
    }
}

impl PreparedRowVersions {
    pub(super) fn append_to(mut self, arrays: &mut Vec<ArrayRef>) -> Result<()> {
        let updated = self
            .columns
            .updated
            .as_ref()
            .map(|source| source.array(self.rows.clone(), &mut self.credit))
            .transpose()?;
        if let Some(array) = &updated {
            arrays.push(array.clone());
        }
        if let Some(source) = &self.columns.created {
            let created = if self.columns.shared_source() {
                updated.ok_or_else(|| Error::internal("missing shared admitted version column"))?
            } else {
                source.array(self.rows, &mut self.credit)?
            };
            arrays.push(created);
        }
        Ok(())
    }
}
