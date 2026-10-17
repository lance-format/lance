// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Selecting one kernel per operation from candidates ranked by preference.
//!
//! ```
//! use lance_core::utils::cpu::Dispatch;
//!
//! type DotFn = unsafe fn(&[f32], &[f32]) -> f32;
//!
//! lance_core::target_kernel! {
//!     #[target_features()]
//!     #[candidate(DOT_PORTABLE: DotFn)]
//!     fn dot_portable(x: &[f32], y: &[f32]) -> f32 {
//!         x.iter().zip(y).map(|(a, b)| a * b).sum()
//!     }
//! }
//!
//! lance_core::dispatch! {
//!     /// Candidates best first; the last must need no features.
//!     static DOT: Dispatch<DotFn> = &[DOT_PORTABLE];
//! }
//!
//! assert_eq!(DOT.get().call(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
//! ```

use std::sync::OnceLock;

use super::CpuFeatures;

/// A kernel together with the CPU features it needs.
///
/// Construct one with [`target_kernel!`](crate::target_kernel), which derives
/// the features from the kernel's own `#[target_feature]` list.
#[derive(Clone, Copy)]
pub struct Candidate<F> {
    name: &'static str,
    features: CpuFeatures,
    kernel: F,
}

impl<F: Copy> Candidate<F> {
    /// # Safety
    ///
    /// Calling `kernel` must be sound on every host that supports `features`.
    pub const unsafe fn new(name: &'static str, features: CpuFeatures, kernel: F) -> Self {
        Self {
            name,
            features,
            kernel,
        }
    }

    /// The kernel's function name.
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Features the host needs to run this kernel.
    pub const fn features(&self) -> CpuFeatures {
        self.features
    }

    /// The kernel, if the host supports it. Lets tests run every candidate.
    pub fn kernel(&self) -> Option<Kernel<F>> {
        CpuFeatures::host()
            .contains(self.features)
            .then_some(Kernel(self.kernel))
    }
}

/// A kernel the host is known to support, so calling it is safe.
#[derive(Clone, Copy)]
pub struct Kernel<F>(F);

impl<A: ?Sized, B: ?Sized, R> Kernel<unsafe fn(&A, &B) -> R> {
    #[inline(always)]
    pub fn call(self, a: &A, b: &B) -> R {
        // SAFETY: a `Kernel` only exists for a candidate whose features the
        // host supports, which is the kernel's only precondition.
        unsafe { (self.0)(a, b) }
    }
}

impl<A: ?Sized, B: ?Sized, C, D: ?Sized, R> Kernel<unsafe fn(&A, &B, C, &mut D) -> R> {
    #[inline(always)]
    pub fn call(self, a: &A, b: &B, c: C, d: &mut D) -> R {
        // SAFETY: see the two-argument `call`.
        unsafe { (self.0)(a, b, c, d) }
    }
}

impl<A: ?Sized, B, C: ?Sized, R> Kernel<unsafe fn(&A, B, &C) -> R> {
    #[inline(always)]
    pub fn call(self, a: &A, b: B, c: &C) -> R {
        // SAFETY: see the two-argument `call`.
        unsafe { (self.0)(a, b, c) }
    }
}

impl<A: ?Sized, B, C, D: ?Sized, R> Kernel<unsafe fn(&A, B, C, &D) -> R> {
    #[inline(always)]
    pub fn call(self, a: &A, b: B, c: C, d: &D) -> R {
        // SAFETY: see the two-argument `call`.
        unsafe { (self.0)(a, b, c, d) }
    }
}

/// One operation's kernels, best first, with the host's choice made once.
///
/// Declare with [`dispatch!`](crate::dispatch). When the build baseline already
/// supports the best candidate, [`Self::get`] returns it without touching the
/// host feature set, so the compiler sees a direct call it can inline.
pub struct Dispatch<F: 'static> {
    candidates: &'static [Candidate<F>],
    // Kept outside `Dispatch` so a `static Dispatch` has no interior
    // mutability; the compiler can then constant-fold `is_static` and the
    // first kernel in `get`.
    selected: &'static Selection<F>,
    is_static: bool,
}

/// Where a [`Dispatch`] caches its host-selected kernel.
///
/// Only [`Dispatch::get`] writes it, and only with a kernel the host supports,
/// so safe code cannot plant a kernel that skips the host check.
pub struct Selection<F>(OnceLock<F>);

impl<F> Selection<F> {
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        Self(OnceLock::new())
    }
}

impl<F: Copy> Dispatch<F> {
    /// Prefer [`dispatch!`](crate::dispatch), which provides `selected`.
    ///
    /// # Panics
    ///
    /// Panics if `candidates` is empty or its last candidate needs any
    /// feature, so that every host has a kernel. Evaluated in a `static`, this
    /// is a compile error.
    pub const fn new(candidates: &'static [Candidate<F>], selected: &'static Selection<F>) -> Self {
        let Some(fallback) = candidates.last() else {
            panic!("a Dispatch needs at least one candidate");
        };
        assert!(
            fallback.features.is_empty(),
            "the last Dispatch candidate must not need any CPU feature"
        );
        Self {
            candidates,
            selected,
            is_static: CpuFeatures::baseline().contains(candidates[0].features),
        }
    }

    /// The best kernel the host supports.
    #[inline(always)]
    pub fn get(&self) -> Kernel<F> {
        if self.is_static {
            return Kernel(self.candidates[0].kernel);
        }
        Kernel(*self.selected.0.get_or_init(|| self.selected().kernel))
    }

    /// The candidate [`Self::get`] returns.
    pub fn selected(&self) -> &'static Candidate<F> {
        &self.candidates[select(self.candidates, CpuFeatures::host())]
    }

    /// Whether the build baseline supports the best candidate, so selection
    /// happens at compile time.
    pub const fn is_static(&self) -> bool {
        self.is_static
    }

    /// All candidates, best first.
    pub const fn candidates(&self) -> &'static [Candidate<F>] {
        self.candidates
    }
}

/// Index of the first candidate `host` supports.
fn select<F>(candidates: &[Candidate<F>], host: CpuFeatures) -> usize {
    candidates
        .iter()
        .position(|candidate| host.contains(candidate.features))
        .expect("Dispatch::new requires a candidate without features")
}

/// Declares a kernel and a [`Candidate`] for it from one feature list.
///
/// The kernel gets one `#[target_feature(enable = ...)]` per listed feature,
/// and the candidate requires exactly those features plus the ones rustc
/// implies, so the two cannot drift apart. The kernel is a safe function: its
/// only precondition is the target features, and a [`Dispatch`] or
/// [`Candidate::kernel`] checks those before handing out a callable
/// [`Kernel`]. Unknown feature names fail to compile.
///
/// The candidate type must be the `unsafe fn` pointer type of the kernel.
///
/// The kernel takes only doc comments and `#[inline]`. Arbitrary attributes
/// are rejected because a `#[target_feature]` among them would enable
/// features the candidate does not require.
#[macro_export]
macro_rules! target_kernel {
    (
        #[target_features($($feature:literal),* $(,)?)]
        #[candidate($candidate_vis:vis $candidate:ident: $kernel_type:ty)]
        $(#[doc = $doc:literal])*
        $(#[inline $(($inline:ident))?])?
        $vis:vis fn $name:ident($($params:tt)*) $(-> $ret:ty)? $body:block
    ) => {
        $(#[doc = $doc])*
        $(#[inline $(($inline))?])?
        $(#[target_feature(enable = $feature)])*
        $vis fn $name($($params)*) $(-> $ret)? $body

        $candidate_vis const $candidate: $crate::utils::cpu::Candidate<$kernel_type> = {
            const FEATURES: $crate::utils::cpu::CpuFeatures =
                $crate::utils::cpu::CpuFeatures::from_target_features(&[$($feature),*]);
            // SAFETY: the kernel is a safe function whose `#[target_feature]`
            // attributes come from the same list as `FEATURES`.
            unsafe { $crate::utils::cpu::Candidate::new(stringify!($name), FEATURES, $name) }
        };
    };
}

/// Declares a `static` [`Dispatch`] over a slice of [`Candidate`]s, best first.
///
/// The slice may be a block whose tail expressions are `#[cfg]`-gated per
/// architecture.
#[macro_export]
macro_rules! dispatch {
    (
        $(#[$attr:meta])*
        $vis:vis static $name:ident: Dispatch<$kernel_type:ty> = $candidates:expr;
    ) => {
        $(#[$attr])*
        $vis static $name: $crate::utils::cpu::Dispatch<$kernel_type> = {
            static SELECTED: $crate::utils::cpu::Selection<$kernel_type> =
                $crate::utils::cpu::Selection::new();
            $crate::utils::cpu::Dispatch::new($candidates, &SELECTED)
        };
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::cpu::CpuFeatures;

    type SumFn = unsafe fn(&[f32], &[f32]) -> f32;
    type FillFn = unsafe fn(&[f32], &[f32], usize, &mut [f32]);

    fn portable_sum(x: &[f32], y: &[f32]) -> f32 {
        x.iter().chain(y).sum()
    }

    fn doubled_sum(x: &[f32], y: &[f32]) -> f32 {
        2.0 * portable_sum(x, y)
    }

    crate::target_kernel! {
        #[target_features()]
        #[candidate(PORTABLE_SUM: SumFn)]
        fn macro_sum(x: &[f32], y: &[f32]) -> f32 {
            portable_sum(x, y)
        }
    }

    crate::target_kernel! {
        #[target_features()]
        #[candidate(PORTABLE_FILL: FillFn)]
        fn macro_fill(x: &[f32], batch: &[f32], dimension: usize, out: &mut [f32]) {
            for (o, y) in out.iter_mut().zip(batch.chunks_exact(dimension)) {
                *o = portable_sum(x, y);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    crate::target_kernel! {
        #[target_features("avx2", "fma")]
        #[candidate(ARCH_SUM: SumFn)]
        fn arch_sum(x: &[f32], y: &[f32]) -> f32 {
            portable_sum(x, y)
        }
    }

    #[cfg(target_arch = "aarch64")]
    crate::target_kernel! {
        #[target_features("neon")]
        #[candidate(ARCH_SUM: SumFn)]
        fn arch_sum(x: &[f32], y: &[f32]) -> f32 {
            portable_sum(x, y)
        }
    }

    fn feature_missing_from_host() -> CpuFeatures {
        let host = CpuFeatures::host();
        CpuFeatures::all()
            .find(|feature| !host.contains(*feature))
            .expect("every host lacks the features of the other architectures")
    }

    #[test]
    fn select_takes_the_first_candidate_the_host_supports() {
        let missing = feature_missing_from_host();
        // SAFETY: the kernels are plain Rust functions without target features.
        let candidates: [Candidate<SumFn>; 2] = unsafe {
            [
                Candidate::new("unsupported", missing, doubled_sum),
                Candidate::new("portable", CpuFeatures::empty(), portable_sum),
            ]
        };
        assert_eq!(select(&candidates, CpuFeatures::host()), 1);
        assert_eq!(select(&candidates, missing), 0);
    }

    #[test]
    fn unsupported_candidate_yields_no_kernel() {
        // SAFETY: the kernel is a plain Rust function without target features.
        let candidate: Candidate<SumFn> =
            unsafe { Candidate::new("unsupported", feature_missing_from_host(), doubled_sum) };
        assert!(candidate.kernel().is_none());
    }

    crate::dispatch! {
        static PORTABLE_ONLY: Dispatch<SumFn> = &[PORTABLE_SUM];
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    crate::dispatch! {
        static ARCH_FIRST: Dispatch<SumFn> = &[ARCH_SUM, PORTABLE_SUM];
    }

    crate::dispatch! {
        static FILL: Dispatch<FillFn> = &[PORTABLE_FILL];
    }

    #[test]
    fn portable_only_dispatch_is_static_and_callable() {
        assert!(PORTABLE_ONLY.is_static());
        assert_eq!(PORTABLE_ONLY.get().call(&[1.0, 2.0], &[3.0]), 6.0);
        assert_eq!(PORTABLE_ONLY.selected().name(), "macro_sum");
    }

    #[test]
    fn slice_output_kernel_is_callable() {
        let mut out = [0.0; 2];
        FILL.get().call(&[1.0], &[2.0, 3.0], 1, &mut out);
        assert_eq!(out, [3.0, 4.0]);
    }

    type ScaledSumFn = unsafe fn(&[f32], f32, &[f32]) -> f32;
    type DoublyScaledSumFn = unsafe fn(&[f32], f32, f32, &[f32]) -> f32;

    crate::target_kernel! {
        #[target_features()]
        #[candidate(SCALED_SUM: ScaledSumFn)]
        fn scaled_sum(x: &[f32], scale: f32, y: &[f32]) -> f32 {
            scale * portable_sum(x, y)
        }
    }

    crate::target_kernel! {
        #[target_features()]
        #[candidate(DOUBLY_SCALED_SUM: DoublyScaledSumFn)]
        fn doubly_scaled_sum(x: &[f32], a: f32, b: f32, y: &[f32]) -> f32 {
            a * b * portable_sum(x, y)
        }
    }

    crate::dispatch! {
        static SCALED: Dispatch<ScaledSumFn> = &[SCALED_SUM];
    }

    crate::dispatch! {
        static DOUBLY_SCALED: Dispatch<DoublyScaledSumFn> = &[DOUBLY_SCALED_SUM];
    }

    #[test]
    fn scalar_argument_kernels_are_callable() {
        assert_eq!(SCALED.get().call(&[1.0], 2.0, &[3.0]), 8.0);
        assert_eq!(DOUBLY_SCALED.get().call(&[1.0], 2.0, 3.0, &[3.0]), 24.0);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn macro_mask_matches_declared_target_features() {
        #[cfg(target_arch = "x86_64")]
        let names: &[&str] = &["avx2", "fma"];
        #[cfg(target_arch = "aarch64")]
        let names: &[&str] = &["neon"];
        assert_eq!(
            ARCH_SUM.features(),
            CpuFeatures::from_target_features(names)
        );
        assert_eq!(ARCH_SUM.name(), "arch_sum");
        assert!(PORTABLE_SUM.features().is_empty());

        let is_supported = CpuFeatures::host().contains(ARCH_SUM.features());
        assert_eq!(ARCH_FIRST.selected().name() == "arch_sum", is_supported);
        assert_eq!(
            ARCH_FIRST.is_static(),
            CpuFeatures::baseline().contains(ARCH_SUM.features())
        );
        for candidate in ARCH_FIRST.candidates() {
            if let Some(kernel) = candidate.kernel() {
                assert_eq!(kernel.call(&[1.0], &[2.0]), 3.0, "{}", candidate.name());
            }
        }
    }
}
