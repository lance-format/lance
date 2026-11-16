// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Translated addresses in physical partition order, without encoded columns.

use std::borrow::Cow;

use arrow_array::{Array, UInt64Array};
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::deepsize::{Context, DeepSizeOf};

pub struct TranslatedRowAddresses(pub UInt64Array);

impl DeepSizeOf for TranslatedRowAddresses {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        self.0.get_buffer_memory_size()
    }
}

/// The enclosing namespace includes the immutable file location and the
/// segment's translation identity. Snapshot filters never enter this value.
pub struct TranslatedRowAddressesKey(pub usize);

impl CacheKey for TranslatedRowAddressesKey {
    type ValueType = TranslatedRowAddresses;

    fn key(&self) -> Cow<'_, str> {
        self.0.to_string().into()
    }

    fn type_name() -> &'static str {
        "RabitTranslatedRowAddresses"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.rabit-translated-row-addresses", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.0 as u64);
    }
}
