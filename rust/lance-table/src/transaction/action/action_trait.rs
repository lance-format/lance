// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `trait Action`: the per-payload trait every concrete action implements.
//!
//! Each payload struct (`AddFragments`, `RemoveFragments`, …) implements this
//! trait with the `apply` / `validate` body the resolver and translation
//! layers dispatch on. Actions are carried as `Box<dyn Action>` in the action
//! list; the [`as_any`](Action::as_any) downcast handle is the entry point for
//! the handful of slice-shape pattern matches that still need to inspect
//! concrete payload type (e.g. `operation_from_actions`).
//!
//! `validate` defaults to `Ok(())` so most payloads need no structural
//! precondition. Conflict-resolution masks (`reads` / `writes`) land in PR B
//! alongside the resolver that consumes them.

use std::any::Any;
use std::fmt::Debug;

use crate::format::{IndexMetadata, Manifest};
use deepsize::DeepSizeOf;
use lance_core::Result;

/// A single granular change to the manifest, addressable by its concrete
/// payload type.
///
/// `Action` is object-safe (no generic methods, no `Self` in signatures
/// beyond `&self`), so it can be held as `Box<dyn Action>` in the action
/// list. Equality, cloning, and deep-size accounting on the trait object are
/// provided by the `impl_dyn_action!` macro and the blanket impls in this
/// module.
pub trait Action: Send + Sync + Debug + DeepSizeOf {
    /// Erased downcast handle — the entry point for slice-shape pattern
    /// matching that survives the migration (e.g. `operation_from_actions`).
    fn as_any(&self) -> &dyn Any;

    /// Type-aware equality on trait objects. Two actions are equal iff they
    /// have the same concrete type and that type's `PartialEq` says so.
    fn dyn_eq(&self, other: &dyn Action) -> bool;

    /// Clone behind the trait object. Each impl forwards to the payload's
    /// `Clone` and re-boxes the result.
    fn dyn_clone(&self) -> Box<dyn Action>;

    /// Apply this action to a manifest in place.
    ///
    /// `indices` is the dataset's secondary-index list — index-domain actions
    /// mutate it; fragment- and schema-domain actions leave it alone.
    fn apply(&self, manifest: &mut Manifest, indices: &mut Vec<IndexMetadata>) -> Result<()>;

    /// Structural pre-screen against `manifest`. Defaults to `Ok(())` — most
    /// payloads have no manifest-structural precondition; the handful that do
    /// (e.g. `AddFields`, `ReplaceFragmentColumns`) override.
    fn validate(&self, _manifest: &Manifest) -> Result<()> {
        Ok(())
    }
}

/// Generate the trait-object boilerplate (`as_any` / `dyn_eq` / `dyn_clone`)
/// for a payload struct.
///
/// Use inside the `impl Action for $t` block. Relies on the payload
/// implementing `PartialEq` and `Clone`.
#[macro_export]
macro_rules! impl_dyn_action {
    ($t:ty) => {
        fn as_any(&self) -> &dyn ::std::any::Any {
            self
        }
        fn dyn_eq(&self, other: &dyn $crate::transaction::action::Action) -> bool {
            other
                .as_any()
                .downcast_ref::<$t>()
                .is_some_and(|o| self == o)
        }
        fn dyn_clone(&self) -> ::std::boxed::Box<dyn $crate::transaction::action::Action> {
            ::std::boxed::Box::new(<$t as ::std::clone::Clone>::clone(self))
        }
    };
}

impl PartialEq for dyn Action {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}

impl Clone for Box<dyn Action> {
    fn clone(&self) -> Self {
        self.dyn_clone()
    }
}

// `Box<T: ?Sized + DeepSizeOf>` already has a blanket `DeepSizeOf` impl in
// `deepsize`, so `Box<dyn Action>` picks it up automatically.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::action::test_support::*;
    use crate::transaction::action::{AddFragments, FragmentSelector, RemoveFragments};

    #[test]
    fn dyn_eq_returns_true_for_same_type_and_value() {
        let a: Box<dyn Action> = Box::new(AddFragments {
            fragments: vec![sample_fragment(0)],
            inserted_rows_filter: None,
        });
        let b: Box<dyn Action> = Box::new(AddFragments {
            fragments: vec![sample_fragment(0)],
            inserted_rows_filter: None,
        });
        assert!(a.dyn_eq(&*b));
        // PartialEq for dyn Action goes through dyn_eq.
        assert_eq!(&*a, &*b);
    }

    #[test]
    fn dyn_eq_returns_false_for_different_types() {
        let add: Box<dyn Action> = Box::new(AddFragments {
            fragments: vec![],
            inserted_rows_filter: None,
        });
        let remove: Box<dyn Action> = Box::new(RemoveFragments {
            selector: FragmentSelector::Ids(vec![1]),
        });
        assert!(!add.dyn_eq(&*remove));
    }

    #[test]
    fn dyn_clone_preserves_concrete_type_and_value() {
        let original: Box<dyn Action> = Box::new(RemoveFragments {
            selector: FragmentSelector::Ids(vec![1, 2, 3]),
        });
        let cloned = original.clone();
        assert!(original.dyn_eq(&*cloned));
        let downcast = cloned.as_any().downcast_ref::<RemoveFragments>().unwrap();
        assert_eq!(downcast.selector, FragmentSelector::Ids(vec![1, 2, 3]));
    }

    #[test]
    fn as_any_supports_downcast() {
        let action: Box<dyn Action> = Box::new(AddFragments {
            fragments: vec![sample_fragment(42)],
            inserted_rows_filter: None,
        });
        let payload = action
            .as_any()
            .downcast_ref::<AddFragments>()
            .expect("downcast to AddFragments");
        assert_eq!(payload.fragments[0].id, 42);
    }

    #[test]
    fn deep_size_of_box_dyn_is_nonzero_and_grows_with_payload() {
        // The blanket `Box<T: ?Sized + DeepSizeOf>` impl carries through to
        // `Box<dyn Action>`; assert the boxed deep size scales with the
        // payload's heap-resident content rather than pinning an exact size
        // (which is fragile across Fragment field changes).
        let small: Box<dyn Action> = Box::new(AddFragments {
            fragments: vec![],
            inserted_rows_filter: None,
        });
        let large: Box<dyn Action> = Box::new(AddFragments {
            fragments: vec![sample_fragment(0), sample_fragment(1)],
            inserted_rows_filter: None,
        });
        assert!(large.deep_size_of() > small.deep_size_of());
    }
}
