// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! CPU feature detection and runtime kernel selection.
//!
//! [`CpuFeatures`] is the set of features the host supports, detected once.
//! Kernels declare the features they need with [`target_kernel!`](crate::target_kernel),
//! and a [`Dispatch`] picks the first candidate the host supports.

use std::fmt;
use std::sync::LazyLock;

mod dispatch;

pub use dispatch::{Candidate, Dispatch, Kernel, Selection};

/// A set of CPU features, named after rustc `target_feature`s.
///
/// Unlike [`SimdSupport`], this is not a ladder: a kernel can require any
/// combination, such as AVX-512 without FMA or F16C on an AVX host.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CpuFeatures(u32);

impl CpuFeatures {
    pub const SSE2: Self = Self(1 << 0);
    pub const SSE3: Self = Self(1 << 1);
    pub const SSSE3: Self = Self(1 << 2);
    pub const SSE4_1: Self = Self(1 << 3);
    pub const SSE4_2: Self = Self(1 << 4);
    pub const POPCNT: Self = Self(1 << 5);
    pub const AVX: Self = Self(1 << 6);
    pub const AVX2: Self = Self(1 << 7);
    pub const FMA: Self = Self(1 << 8);
    pub const F16C: Self = Self(1 << 9);
    pub const BMI1: Self = Self(1 << 10);
    pub const BMI2: Self = Self(1 << 11);
    pub const AVX512F: Self = Self(1 << 12);
    pub const AVX512BW: Self = Self(1 << 13);
    pub const AVX512CD: Self = Self(1 << 14);
    pub const AVX512DQ: Self = Self(1 << 15);
    pub const AVX512VL: Self = Self(1 << 16);
    pub const AVX512FP16: Self = Self(1 << 17);
    pub const NEON: Self = Self(1 << 18);
    pub const FP16: Self = Self(1 << 19);
    pub const LSX: Self = Self(1 << 20);
    pub const LASX: Self = Self(1 << 21);
    pub const AVX512VNNI: Self = Self(1 << 22);
    pub const AVX512VPOPCNTDQ: Self = Self(1 << 23);
    pub const AVX512VBMI: Self = Self(1 << 24);
    pub const AVX512BF16: Self = Self(1 << 25);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every feature in `other` is also in `self`.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Features supported by the host, detected once per process.
    pub fn host() -> Self {
        *HOST_FEATURES
    }

    /// Features the build enables for all code, e.g. via `-C target-cpu`.
    ///
    /// A kernel whose features are all in the baseline can be called directly
    /// and inlined; see [`Dispatch`].
    pub const fn baseline() -> Self {
        BASELINE
    }

    /// The feature set for one rustc `target_feature` name, including the
    /// features rustc enables along with it. `None` if the name is unknown.
    ///
    /// Implied features matter for soundness: `#[target_feature(enable =
    /// "avx512f")]` also lets codegen use AVX2, FMA and F16C, so a kernel
    /// declaring it must not run on a host that lacks any of them.
    pub const fn from_target_feature(name: &str) -> Option<Self> {
        let mut i = 0;
        while i < FEATURE_TABLE.len() {
            if str_eq(FEATURE_TABLE[i].name, name) {
                return Some(FEATURE_TABLE[i].feature.with_implied());
            }
            i += 1;
        }
        None
    }

    /// Union of [`Self::from_target_feature`] over `names`.
    ///
    /// # Panics
    ///
    /// Panics if a name is unknown. [`target_kernel!`](crate::target_kernel)
    /// evaluates this in a `const`, so a typo is a compile error there.
    pub const fn from_target_features(names: &[&str]) -> Self {
        let mut features = Self::empty();
        let mut i = 0;
        while i < names.len() {
            match Self::from_target_feature(names[i]) {
                Some(feature) => features = features.union(feature),
                None => panic!("unknown target feature; add it to CpuFeatures"),
            }
            i += 1;
        }
        features
    }

    /// Every known feature, one per set.
    pub fn all() -> impl Iterator<Item = Self> {
        FEATURE_TABLE.iter().map(|info| info.feature)
    }

    /// The `target_feature` names of the features in this set.
    pub fn names(self) -> impl Iterator<Item = &'static str> {
        FEATURE_TABLE
            .iter()
            .filter(move |info| self.contains(info.feature))
            .map(|info| info.name)
    }

    fn x86_names(self) -> impl Iterator<Item = &'static str> {
        FEATURE_TABLE
            .iter()
            .filter(move |info| info.is_x86 && self.contains(info.feature))
            .map(|info| info.name)
    }

    const fn with_implied(self) -> Self {
        let mut features = self;
        loop {
            let mut expanded = features;
            let mut i = 0;
            while i < FEATURE_TABLE.len() {
                if features.contains(FEATURE_TABLE[i].feature) {
                    expanded = expanded.union(FEATURE_TABLE[i].implies);
                }
                i += 1;
            }
            if expanded.0 == features.0 {
                return features;
            }
            features = expanded;
        }
    }
}

impl fmt::Debug for CpuFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.names()).finish()
    }
}

struct FeatureInfo {
    feature: CpuFeatures,
    name: &'static str,
    is_x86: bool,
    /// Features rustc enables along with this one. Transitive implications
    /// are resolved by [`CpuFeatures::with_implied`].
    implies: CpuFeatures,
}

const fn x86(feature: CpuFeatures, name: &'static str, implies: CpuFeatures) -> FeatureInfo {
    FeatureInfo {
        feature,
        name,
        is_x86: true,
        implies,
    }
}

const fn other(feature: CpuFeatures, name: &'static str, implies: CpuFeatures) -> FeatureInfo {
    FeatureInfo {
        feature,
        name,
        is_x86: false,
        implies,
    }
}

/// Ordered as [`SimdInfo::host_features`] reports them.
const FEATURE_TABLE: &[FeatureInfo] = {
    use CpuFeatures as F;
    &[
        x86(F::SSE2, "sse2", F::empty()),
        x86(F::SSE3, "sse3", F::SSE2),
        x86(F::SSSE3, "ssse3", F::SSE3),
        x86(F::SSE4_1, "sse4.1", F::SSSE3),
        x86(F::SSE4_2, "sse4.2", F::SSE4_1),
        x86(F::POPCNT, "popcnt", F::empty()),
        x86(F::AVX, "avx", F::SSE4_2),
        x86(F::AVX2, "avx2", F::AVX),
        x86(F::FMA, "fma", F::AVX),
        x86(F::F16C, "f16c", F::AVX),
        x86(F::BMI1, "bmi1", F::empty()),
        x86(F::BMI2, "bmi2", F::empty()),
        x86(F::AVX512F, "avx512f", F::AVX2.union(F::FMA).union(F::F16C)),
        x86(F::AVX512BW, "avx512bw", F::AVX512F),
        x86(F::AVX512CD, "avx512cd", F::AVX512F),
        x86(F::AVX512DQ, "avx512dq", F::AVX512F),
        x86(F::AVX512VL, "avx512vl", F::AVX512F),
        x86(
            F::AVX512FP16,
            "avx512fp16",
            F::AVX512BW.union(F::AVX512DQ).union(F::AVX512VL),
        ),
        x86(F::AVX512VNNI, "avx512vnni", F::AVX512F),
        x86(F::AVX512VPOPCNTDQ, "avx512vpopcntdq", F::AVX512F),
        x86(F::AVX512VBMI, "avx512vbmi", F::AVX512BW),
        x86(F::AVX512BF16, "avx512bf16", F::AVX512BW),
        other(F::NEON, "neon", F::empty()),
        other(F::FP16, "fp16", F::NEON),
        other(F::LSX, "lsx", F::empty()),
        other(F::LASX, "lasx", F::LSX),
    ]
};

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const BASELINE: CpuFeatures = {
    // `cfg!` needs a literal feature name, so this cannot loop over the table.
    let enabled = [
        (cfg!(target_feature = "sse2"), CpuFeatures::SSE2),
        (cfg!(target_feature = "sse3"), CpuFeatures::SSE3),
        (cfg!(target_feature = "ssse3"), CpuFeatures::SSSE3),
        (cfg!(target_feature = "sse4.1"), CpuFeatures::SSE4_1),
        (cfg!(target_feature = "sse4.2"), CpuFeatures::SSE4_2),
        (cfg!(target_feature = "popcnt"), CpuFeatures::POPCNT),
        (cfg!(target_feature = "avx"), CpuFeatures::AVX),
        (cfg!(target_feature = "avx2"), CpuFeatures::AVX2),
        (cfg!(target_feature = "fma"), CpuFeatures::FMA),
        (cfg!(target_feature = "f16c"), CpuFeatures::F16C),
        (cfg!(target_feature = "bmi1"), CpuFeatures::BMI1),
        (cfg!(target_feature = "bmi2"), CpuFeatures::BMI2),
        (cfg!(target_feature = "avx512f"), CpuFeatures::AVX512F),
        (cfg!(target_feature = "avx512bw"), CpuFeatures::AVX512BW),
        (cfg!(target_feature = "avx512cd"), CpuFeatures::AVX512CD),
        (cfg!(target_feature = "avx512dq"), CpuFeatures::AVX512DQ),
        (cfg!(target_feature = "avx512vl"), CpuFeatures::AVX512VL),
        (cfg!(target_feature = "avx512fp16"), CpuFeatures::AVX512FP16),
        (cfg!(target_feature = "avx512vnni"), CpuFeatures::AVX512VNNI),
        (
            cfg!(target_feature = "avx512vpopcntdq"),
            CpuFeatures::AVX512VPOPCNTDQ,
        ),
        (cfg!(target_feature = "avx512vbmi"), CpuFeatures::AVX512VBMI),
        (cfg!(target_feature = "avx512bf16"), CpuFeatures::AVX512BF16),
        (cfg!(target_feature = "neon"), CpuFeatures::NEON),
        (cfg!(target_feature = "fp16"), CpuFeatures::FP16),
        (cfg!(target_feature = "lsx"), CpuFeatures::LSX),
        (cfg!(target_feature = "lasx"), CpuFeatures::LASX),
    ];
    let mut features = CpuFeatures::empty();
    let mut i = 0;
    while i < enabled.len() {
        if enabled[i].0 {
            features = features.union(enabled[i].1);
        }
        i += 1;
    }
    features
};

static HOST_FEATURES: LazyLock<CpuFeatures> =
    LazyLock::new(|| detect_host_features().union(BASELINE));

#[cfg(target_arch = "x86_64")]
fn detect_host_features() -> CpuFeatures {
    let mut features = CpuFeatures::empty();
    let mut add = |is_detected: bool, feature: CpuFeatures| {
        if is_detected {
            features = features.union(feature);
        }
    };
    // Each call must be inline: `is_x86_feature_detected!` does its own custom
    // input parsing and rejects feature names received via a `macro_rules!`
    // `:literal` metavariable on some toolchains.
    add(is_x86_feature_detected!("sse2"), CpuFeatures::SSE2);
    add(is_x86_feature_detected!("sse3"), CpuFeatures::SSE3);
    add(is_x86_feature_detected!("ssse3"), CpuFeatures::SSSE3);
    add(is_x86_feature_detected!("sse4.1"), CpuFeatures::SSE4_1);
    add(is_x86_feature_detected!("sse4.2"), CpuFeatures::SSE4_2);
    add(is_x86_feature_detected!("popcnt"), CpuFeatures::POPCNT);
    add(is_x86_feature_detected!("avx"), CpuFeatures::AVX);
    add(is_x86_feature_detected!("avx2"), CpuFeatures::AVX2);
    add(is_x86_feature_detected!("fma"), CpuFeatures::FMA);
    add(is_x86_feature_detected!("f16c"), CpuFeatures::F16C);
    add(is_x86_feature_detected!("bmi1"), CpuFeatures::BMI1);
    add(is_x86_feature_detected!("bmi2"), CpuFeatures::BMI2);
    add(is_x86_feature_detected!("avx512f"), CpuFeatures::AVX512F);
    add(is_x86_feature_detected!("avx512bw"), CpuFeatures::AVX512BW);
    add(is_x86_feature_detected!("avx512cd"), CpuFeatures::AVX512CD);
    add(is_x86_feature_detected!("avx512dq"), CpuFeatures::AVX512DQ);
    add(is_x86_feature_detected!("avx512vl"), CpuFeatures::AVX512VL);
    add(
        is_x86_feature_detected!("avx512fp16"),
        CpuFeatures::AVX512FP16,
    );
    add(
        is_x86_feature_detected!("avx512vnni"),
        CpuFeatures::AVX512VNNI,
    );
    add(
        is_x86_feature_detected!("avx512vpopcntdq"),
        CpuFeatures::AVX512VPOPCNTDQ,
    );
    add(
        is_x86_feature_detected!("avx512vbmi"),
        CpuFeatures::AVX512VBMI,
    );
    add(
        is_x86_feature_detected!("avx512bf16"),
        CpuFeatures::AVX512BF16,
    );
    features
}

#[cfg(target_arch = "aarch64")]
fn detect_host_features() -> CpuFeatures {
    #[cfg(any(target_os = "ios", target_os = "tvos"))]
    let has_fp16 = true;
    #[cfg(not(any(target_os = "ios", target_os = "tvos")))]
    let has_fp16 = aarch64::has_neon_f16_support();

    if has_fp16 {
        CpuFeatures::NEON.union(CpuFeatures::FP16)
    } else {
        CpuFeatures::NEON
    }
}

#[cfg(target_arch = "loongarch64")]
fn detect_host_features() -> CpuFeatures {
    let mut features = CpuFeatures::empty();
    if loongarch64::has_lsx_support() {
        features = features.union(CpuFeatures::LSX);
    }
    if loongarch64::has_lasx_support() {
        features = features.union(CpuFeatures::LASX);
    }
    features
}

#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "loongarch64"
)))]
fn detect_host_features() -> CpuFeatures {
    CpuFeatures::empty()
}

/// A level of SIMD support for some feature.
///
/// `#[non_exhaustive]` so future tiers (e.g. AVX-512 BF16, AMX) can be added
/// without breaking external `match` consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SimdSupport {
    None,
    Neon,
    Sse,
    /// AVX (256-bit float ops) without FMA.
    ///
    /// This tier does not imply that AVX2 is absent: selecting [`Self::Avx2`]
    /// requires both AVX2 and FMA, so a host with AVX2 but no FMA selects this
    /// tier. Intel Sandy Bridge / Ivy Bridge are the typical hosts.
    Avx,
    /// AVX + FMA but no AVX2.
    /// AMD Piledriver / Steamroller / FX-7500.
    AvxFma,
    /// AVX2 + FMA. Intel Haswell / AMD Excavator and later.
    ///
    /// Selecting this tier asserts FMA is present: the kernels it dispatches to
    /// are `#[target_feature(enable = "avx,fma")]`.
    Avx2,
    Avx512,
    Avx512FP16,
    Lsx,
    Lasx,
}

impl fmt::Display for SimdSupport {
    /// Formats the tier name in lowercase, matching pyarrow's
    /// `runtime_info().simd_level` convention.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::None => "none",
            Self::Neon => "neon",
            Self::Sse => "sse",
            Self::Avx => "avx",
            Self::AvxFma => "avx_fma",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
            Self::Avx512FP16 => "avx512_fp16",
            Self::Lsx => "lsx",
            Self::Lasx => "lasx",
        };
        f.write_str(name)
    }
}

/// Snapshot of the SIMD tier lance dispatches to on the current host, plus the
/// raw CPU features detected for diagnostic purposes.
///
/// Mirrors the role of `pyarrow.runtime_info()`: a single, cheap call users can
/// make to verify which SIMD tier the runtime selected and what underlying
/// features the host advertises. Obtain one with [`simd_info()`].
#[derive(Debug, Clone)]
pub struct SimdInfo {
    /// The SIMD tier lance dispatches to at runtime on this host.
    pub tier: SimdSupport,
    /// The architecture name (e.g. "x86_64", "aarch64", "loongarch64").
    pub target_arch: &'static str,
    /// Raw CPU feature flags detected on this host (x86_64 only; empty on
    /// other architectures). Each entry is a feature name like "avx2",
    /// "fma", "avx512f", "popcnt", etc.
    pub host_features: Vec<&'static str>,
}

/// Returns a snapshot of the SIMD tier lance is using on this host along with
/// the raw CPU feature flags that drove the decision.
///
/// Useful for performance debugging and giving users a way to verify which
/// dispatch tier they are hitting without rebuilding lance. See [`SimdInfo`]
/// for the meaning of each field and [`SimdSupport`] for the tier values.
///
/// # Examples
///
/// ```
/// use lance_core::utils::cpu::simd_info;
///
/// let info = simd_info();
/// println!("dispatching to {} on {}", info.tier, info.target_arch);
/// ```
pub fn simd_info() -> SimdInfo {
    SimdInfo {
        tier: *SIMD_SUPPORT,
        target_arch: std::env::consts::ARCH,
        host_features: CpuFeatures::host().x86_names().collect(),
    }
}

impl SimdSupport {
    /// The tier lance reports for a feature set. Each architecture only sets
    /// its own feature bits, so checking every architecture in turn is safe.
    fn from_features(features: CpuFeatures) -> Self {
        use CpuFeatures as F;
        if features.contains(F::AVX512F) {
            if features.contains(F::AVX512FP16) {
                Self::Avx512FP16
            } else {
                Self::Avx512
            }
        } else if features.contains(F::AVX2.union(F::FMA)) {
            Self::Avx2
        } else if features.contains(F::AVX.union(F::FMA)) {
            // AMD Piledriver / Steamroller / FX-7500: 256-bit float ops + FMA but no AVX2.
            Self::AvxFma
        } else if features.contains(F::AVX) {
            // This includes a possible AVX2 host without FMA because the Avx2
            // tier above requires both features.
            Self::Avx
        } else if features.contains(F::FP16) {
            Self::Neon
        } else if features.contains(F::LASX) {
            Self::Lasx
        } else if features.contains(F::LSX) {
            Self::Lsx
        } else {
            Self::None
        }
    }
}

/// Support for SIMD operations, derived from [`CpuFeatures::host`].
pub static SIMD_SUPPORT: LazyLock<SimdSupport> =
    LazyLock::new(|| SimdSupport::from_features(CpuFeatures::host()));

// Inspired by https://github.com/RustCrypto/utils/blob/master/cpufeatures/src/aarch64.rs
// aarch64 doesn't have userspace feature detection built in, so we have to call
// into OS-specific functions to check for features.

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod aarch64 {
    pub fn has_neon_f16_support() -> bool {
        // Maybe we can assume it's there?
        true
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
mod aarch64 {
    pub fn has_neon_f16_support() -> bool {
        // See: https://github.com/rust-lang/libc/blob/7ce81ca7aeb56aae7ca0237ef9353d58f3d7d2f1/src/unix/linux_like/linux/gnu/b64/aarch64/mod.rs#L533
        let flags = unsafe { libc::getauxval(libc::AT_HWCAP) };
        flags & libc::HWCAP_FPHP != 0
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "windows"))]
mod aarch64 {
    pub fn has_neon_f16_support() -> bool {
        // https://github.com/lance-format/lance/issues/2411
        false
    }
}

#[cfg(target_arch = "loongarch64")]
mod loongarch64 {
    pub fn has_lsx_support() -> bool {
        // See: https://github.com/rust-lang/libc/blob/7ce81ca7aeb56aae7ca0237ef9353d58f3d7d2f1/src/unix/linux_like/linux/gnu/b64/loongarch64/mod.rs#L263
        let flags = unsafe { libc::getauxval(libc::AT_HWCAP) };
        flags & libc::HWCAP_LOONGARCH_LSX != 0
    }
    pub fn has_lasx_support() -> bool {
        // See: https://github.com/rust-lang/libc/blob/7ce81ca7aeb56aae7ca0237ef9353d58f3d7d2f1/src/unix/linux_like/linux/gnu/b64/loongarch64/mod.rs#L264
        let flags = unsafe { libc::getauxval(libc::AT_HWCAP) };
        flags & libc::HWCAP_LOONGARCH_LASX != 0
    }
}

#[cfg(all(target_arch = "aarch64", target_os = "android"))]
mod aarch64 {
    pub fn has_neon_f16_support() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn simd_info_exposes_tier() {
        let info = simd_info();
        assert_eq!(info.target_arch, std::env::consts::ARCH);
        // Tier should match the detected SIMD support.
        assert_eq!(info.tier, *SIMD_SUPPORT);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_info_features_include_baseline() {
        let info = simd_info();
        // The x86_64 ABI mandates SSE2, so it must always be present on this
        // architecture.
        assert!(info.host_features.contains(&"sse2"));
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn simd_info_features_empty_off_x86_64() {
        let info = simd_info();
        assert!(info.host_features.is_empty());
    }

    /// The `Avx2` and `AvxFma` tiers both dispatch to kernels declared
    /// `#[target_feature(enable = "avx,fma")]`, so neither may be selected on a
    /// host without FMA. AVX2 does not imply FMA in the ISA, so the detection
    /// checks it explicitly.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx_fma_tiers_are_only_selected_when_fma_is_detected() {
        if matches!(*SIMD_SUPPORT, SimdSupport::Avx2 | SimdSupport::AvxFma) {
            assert!(
                is_x86_feature_detected!("fma"),
                "tier {} dispatches to avx,fma kernels but the host has no FMA",
                *SIMD_SUPPORT
            );
        }
    }

    #[test]
    fn host_features_include_build_baseline() {
        assert!(CpuFeatures::host().contains(CpuFeatures::baseline()));
    }

    /// The mask a kernel declares must cover every feature rustc enables for
    /// it, including implied ones, or a host lacking an implied feature could
    /// select a kernel that uses it.
    #[rstest]
    #[case::sse_chain("avx", CpuFeatures::SSE2.union(CpuFeatures::SSE4_2))]
    #[case::fma_implies_avx("fma", CpuFeatures::AVX)]
    #[case::avx512f(
        "avx512f",
        CpuFeatures::AVX2.union(CpuFeatures::FMA).union(CpuFeatures::F16C)
    )]
    #[case::avx512fp16("avx512fp16", CpuFeatures::AVX512BW.union(CpuFeatures::AVX512F))]
    #[case::avx512vnni("avx512vnni", CpuFeatures::AVX512F.union(CpuFeatures::FMA))]
    #[case::avx512vpopcntdq("avx512vpopcntdq", CpuFeatures::AVX512F.union(CpuFeatures::F16C))]
    #[case::avx512vbmi("avx512vbmi", CpuFeatures::AVX512BW.union(CpuFeatures::AVX2))]
    #[case::avx512bf16("avx512bf16", CpuFeatures::AVX512BW.union(CpuFeatures::FMA))]
    #[case::fp16("fp16", CpuFeatures::NEON)]
    #[case::lasx("lasx", CpuFeatures::LSX)]
    fn target_feature_mask_includes_implied_features(
        #[case] name: &str,
        #[case] implied: CpuFeatures,
    ) {
        let features = CpuFeatures::from_target_feature(name).unwrap();
        assert!(
            features.contains(implied),
            "{name} -> {features:?} is missing {implied:?}"
        );
    }

    #[test]
    fn features_are_distinct_bits() {
        let mut seen = CpuFeatures::empty();
        for feature in CpuFeatures::all() {
            assert_eq!(feature.0.count_ones(), 1, "{feature:?}");
            assert!(!seen.contains(feature), "{feature:?} reuses a bit");
            seen = seen.union(feature);
        }
    }

    #[test]
    fn unknown_target_feature_is_rejected() {
        assert_eq!(CpuFeatures::from_target_feature("avx9000"), None);
    }

    #[rstest]
    #[case::none(&[], SimdSupport::None)]
    #[case::avx(&["avx"], SimdSupport::Avx)]
    #[case::avx2_without_fma(&["avx2"], SimdSupport::Avx)]
    #[case::avx_fma(&["fma"], SimdSupport::AvxFma)]
    #[case::avx2(&["avx2", "fma"], SimdSupport::Avx2)]
    #[case::avx512(&["avx512f"], SimdSupport::Avx512)]
    #[case::avx512_fp16(&["avx512fp16"], SimdSupport::Avx512FP16)]
    #[case::neon_without_fp16(&["neon"], SimdSupport::None)]
    #[case::neon(&["fp16"], SimdSupport::Neon)]
    #[case::lsx(&["lsx"], SimdSupport::Lsx)]
    #[case::lasx(&["lasx"], SimdSupport::Lasx)]
    fn simd_support_is_derived_from_features(
        #[case] names: &[&str],
        #[case] expected: SimdSupport,
    ) {
        let features = CpuFeatures::from_target_features(names);
        assert_eq!(SimdSupport::from_features(features), expected);
    }

    #[rstest]
    #[case::none(SimdSupport::None, "none")]
    #[case::neon(SimdSupport::Neon, "neon")]
    #[case::sse(SimdSupport::Sse, "sse")]
    #[case::avx(SimdSupport::Avx, "avx")]
    #[case::avx_fma(SimdSupport::AvxFma, "avx_fma")]
    #[case::avx2(SimdSupport::Avx2, "avx2")]
    #[case::avx512(SimdSupport::Avx512, "avx512")]
    #[case::avx512_fp16(SimdSupport::Avx512FP16, "avx512_fp16")]
    #[case::lsx(SimdSupport::Lsx, "lsx")]
    #[case::lasx(SimdSupport::Lasx, "lasx")]
    fn simd_support_display_matches_lowercase_convention(
        #[case] tier: SimdSupport,
        #[case] expected: &str,
    ) {
        assert_eq!(tier.to_string(), expected);
    }
}
