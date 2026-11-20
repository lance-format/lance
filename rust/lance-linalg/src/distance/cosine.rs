// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cosine distance
//!
//! <https://en.wikipedia.org/wiki/Cosine_similarity>
//!
//! `bf16, f16, f32, f64` types are supported.

use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array};
use half::{bf16, f16};

use super::{
    BatchKernel, Dot, Normalize, PairKernel, assert_batch_output_layout, assert_equal_lengths,
    dot::dot,
    norm_l2::{norm_l2, norm_l2_impl},
};
use crate::Result;
#[cfg(not(target_arch = "x86_64"))]
use crate::simd::{FloatSimd, SIMD};

/// Cosine Distance
pub trait Cosine: Dot + Normalize {
    /// Cosine distance between two vectors.
    #[inline]
    fn cosine(x: &[Self], other: &[Self]) -> f32 {
        let x_norm = norm_l2(x);
        Self::cosine_fast(x, x_norm, other)
    }

    /// Fast cosine function, that assumes that the norm of the first vector is already known.
    #[inline]
    fn cosine_fast(x: &[Self], x_norm: f32, y: &[Self]) -> f32 {
        cosine_scalar(x, x_norm, y)
    }

    /// Cosine between two vectors, with the L2 norms of both vectors already known.
    #[inline]
    fn cosine_with_norms(x: &[Self], x_norm: f32, y_norm: f32, y: &[Self]) -> f32 {
        cosine_scalar_fast(x, x_norm, y, y_norm)
    }

    /// Writes the cosine distance from `x` to each `dimension`-sized vector
    /// in `batch` into the matching element of `output`.
    ///
    /// The default computes the norm of `x` once and calls
    /// [`Cosine::cosine_fast`] per vector. `f32` overrides it with one batch
    /// kernel call.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != dimension`, if `batch.len()` is not a multiple of
    /// `dimension`, or if `output.len() != batch.len() / dimension`.
    fn cosine_batch_into(x: &[Self], batch: &[Self], dimension: usize, output: &mut [f32]) {
        assert_batch_output_layout(x.len(), batch.len(), dimension, output.len());
        let x_norm = norm_l2(x);
        for (distance, y) in output.iter_mut().zip(batch.chunks_exact(dimension)) {
            *distance = Self::cosine_fast(x, x_norm, y);
        }
    }
}

impl Cosine for u8 {
    #[inline]
    fn cosine(x: &[Self], other: &[Self]) -> f32 {
        super::cosine_u8::cosine_u8(x, other)
    }
}

/// Cosine distance from `x`, whose L2 norm is `x_norm`, to `y`.
type CosineKernel<T> = unsafe fn(&[T], f32, &[T]) -> f32;

/// Cosine distance between `x` and `y` given both L2 norms, in the argument
/// order of [`Cosine::cosine_with_norms`].
type CosineWithNormsKernel = unsafe fn(&[f32], f32, f32, &[f32]) -> f32;

impl Cosine for bf16 {
    fn cosine_fast(x: &[Self], x_norm: f32, y: &[Self]) -> f32 {
        assert_equal_lengths(x.len(), y.len());
        COSINE_BF16.get().call(x, x_norm, y)
    }
}

impl Cosine for f16 {
    fn cosine_fast(x: &[Self], x_norm: f32, y: &[Self]) -> f32 {
        assert_equal_lengths(x.len(), y.len());
        COSINE_F16.get().call(x, x_norm, y)
    }
}

lance_core::dispatch! {
    static COSINE_F16: Dispatch<CosineKernel<f16>> = &[
        #[cfg(all(kernel_support = "avx512_f16", target_arch = "x86_64"))]
        half_c::COSINE_F16_AVX512,
        #[cfg(all(kernel_support = "half", target_arch = "x86_64"))]
        half_c::COSINE_F16_AVX2,
        #[cfg(all(kernel_support = "half", target_arch = "aarch64"))]
        half_c::COSINE_F16_NEON,
        #[cfg(all(kernel_support = "half", target_arch = "loongarch64"))]
        half_c::COSINE_F16_LASX,
        #[cfg(all(kernel_support = "half", target_arch = "loongarch64"))]
        half_c::COSINE_F16_LSX,
        COSINE_F16_SCALAR,
    ];
}

lance_core::dispatch! {
    static COSINE_BF16: Dispatch<CosineKernel<bf16>> = &[
        #[cfg(all(kernel_support = "avx512_bf16", target_arch = "x86_64"))]
        half_c::COSINE_BF16_AVX512,
        #[cfg(all(kernel_support = "half", target_arch = "x86_64"))]
        half_c::COSINE_BF16_AVX2,
        #[cfg(all(kernel_support = "half", target_arch = "aarch64"))]
        half_c::COSINE_BF16_NEON,
        #[cfg(all(kernel_support = "half", target_arch = "loongarch64"))]
        half_c::COSINE_BF16_LASX,
        #[cfg(all(kernel_support = "half", target_arch = "loongarch64"))]
        half_c::COSINE_BF16_LSX,
        COSINE_BF16_SCALAR,
    ];
}

lance_core::target_kernel! {
    #[target_features()]
    #[candidate(COSINE_F16_SCALAR: CosineKernel<f16>)]
    fn cosine_f16_scalar(x: &[f16], x_norm: f32, y: &[f16]) -> f32 {
        cosine_scalar(x, x_norm, y)
    }
}

lance_core::target_kernel! {
    #[target_features()]
    #[candidate(COSINE_BF16_SCALAR: CosineKernel<bf16>)]
    fn cosine_bf16_scalar(x: &[bf16], x_norm: f32, y: &[bf16]) -> f32 {
        cosine_scalar(x, x_norm, y)
    }
}

/// The `cosine_f16` and `cosine_bf16` functions in f16.c and bf16.c. build.rs
/// compiles each file once per SIMD level, suffixing the symbols with the level.
#[cfg(kernel_support = "half")]
mod half_c {
    use half::{bf16, f16};

    use super::CosineKernel;

    mod ffi {
        use half::{bf16, f16};

        unsafe extern "C" {
            #[cfg(all(kernel_support = "avx512_f16", target_arch = "x86_64"))]
            pub fn cosine_f16_avx512(
                x: *const f16,
                x_norm: f32,
                y: *const f16,
                dimension: u32,
            ) -> f32;
            #[cfg(target_arch = "x86_64")]
            pub fn cosine_f16_avx2(
                x: *const f16,
                x_norm: f32,
                y: *const f16,
                dimension: u32,
            ) -> f32;
            #[cfg(target_arch = "aarch64")]
            pub fn cosine_f16_neon(
                x: *const f16,
                x_norm: f32,
                y: *const f16,
                dimension: u32,
            ) -> f32;
            #[cfg(target_arch = "loongarch64")]
            pub fn cosine_f16_lasx(
                x: *const f16,
                x_norm: f32,
                y: *const f16,
                dimension: u32,
            ) -> f32;
            #[cfg(target_arch = "loongarch64")]
            pub fn cosine_f16_lsx(x: *const f16, x_norm: f32, y: *const f16, dimension: u32)
            -> f32;

            #[cfg(all(kernel_support = "avx512_bf16", target_arch = "x86_64"))]
            pub fn cosine_bf16_avx512(
                x: *const bf16,
                x_norm: f32,
                y: *const bf16,
                dimension: u32,
            ) -> f32;
            #[cfg(target_arch = "x86_64")]
            pub fn cosine_bf16_avx2(
                x: *const bf16,
                x_norm: f32,
                y: *const bf16,
                dimension: u32,
            ) -> f32;
            #[cfg(target_arch = "aarch64")]
            pub fn cosine_bf16_neon(
                x: *const bf16,
                x_norm: f32,
                y: *const bf16,
                dimension: u32,
            ) -> f32;
            #[cfg(target_arch = "loongarch64")]
            pub fn cosine_bf16_lasx(
                x: *const bf16,
                x_norm: f32,
                y: *const bf16,
                dimension: u32,
            ) -> f32;
            #[cfg(target_arch = "loongarch64")]
            pub fn cosine_bf16_lsx(
                x: *const bf16,
                x_norm: f32,
                y: *const bf16,
                dimension: u32,
            ) -> f32;
        }
    }

    half_c_kernels! {
        fn (x: &[f16], x_norm: f32, y: &[f16]) -> f32
            => (x.as_ptr(), x_norm, y.as_ptr(), x.len().min(y.len()) as u32);
        #[cfg(all(kernel_support = "avx512_f16", target_arch = "x86_64"))]
        COSINE_F16_AVX512: CosineKernel<f16> = cosine_f16_avx512("avx512fp16");
        #[cfg(target_arch = "x86_64")]
        COSINE_F16_AVX2: CosineKernel<f16> = cosine_f16_avx2("avx2", "fma", "f16c", "bmi1", "bmi2");
        #[cfg(target_arch = "aarch64")]
        COSINE_F16_NEON: CosineKernel<f16> = cosine_f16_neon("neon", "fp16");
        #[cfg(target_arch = "loongarch64")]
        COSINE_F16_LASX: CosineKernel<f16> = cosine_f16_lasx("lasx");
        #[cfg(target_arch = "loongarch64")]
        COSINE_F16_LSX: CosineKernel<f16> = cosine_f16_lsx("lsx");
    }

    half_c_kernels! {
        fn (x: &[bf16], x_norm: f32, y: &[bf16]) -> f32
            => (x.as_ptr(), x_norm, y.as_ptr(), x.len().min(y.len()) as u32);
        #[cfg(all(kernel_support = "avx512_bf16", target_arch = "x86_64"))]
        COSINE_BF16_AVX512: CosineKernel<bf16> = cosine_bf16_avx512("avx512fp16", "avx512bf16");
        #[cfg(target_arch = "x86_64")]
        COSINE_BF16_AVX2: CosineKernel<bf16> = cosine_bf16_avx2("avx2", "fma", "f16c", "bmi1", "bmi2");
        #[cfg(target_arch = "aarch64")]
        COSINE_BF16_NEON: CosineKernel<bf16> = cosine_bf16_neon("neon", "fp16");
        #[cfg(target_arch = "loongarch64")]
        COSINE_BF16_LASX: CosineKernel<bf16> = cosine_bf16_lasx("lasx");
        #[cfg(target_arch = "loongarch64")]
        COSINE_BF16_LSX: CosineKernel<bf16> = cosine_bf16_lsx("lsx");
    }

    // Each candidate must require what build.rs compiled its object for.
    #[cfg(target_arch = "x86_64")]
    const _: () = {
        use crate::distance::half_c_objects::{HASWELL, same_features};
        assert!(same_features(COSINE_F16_AVX2.features(), HASWELL));
        assert!(same_features(COSINE_BF16_AVX2.features(), HASWELL));
    };
    #[cfg(all(target_arch = "x86_64", kernel_support = "avx512_f16"))]
    const _: () = {
        use crate::distance::half_c_objects::{SAPPHIRE_RAPIDS_F16, same_features};
        assert!(same_features(
            COSINE_F16_AVX512.features(),
            SAPPHIRE_RAPIDS_F16
        ));
    };
    #[cfg(all(target_arch = "x86_64", kernel_support = "avx512_bf16"))]
    const _: () = {
        use crate::distance::half_c_objects::{SAPPHIRE_RAPIDS_BF16, same_features};
        assert!(same_features(
            COSINE_BF16_AVX512.features(),
            SAPPHIRE_RAPIDS_BF16
        ));
    };
    #[cfg(target_arch = "aarch64")]
    const _: () = {
        use crate::distance::half_c_objects::{ARMV8_2_FP16, same_features};
        assert!(same_features(COSINE_F16_NEON.features(), ARMV8_2_FP16));
        assert!(same_features(COSINE_BF16_NEON.features(), ARMV8_2_FP16));
    };
    #[cfg(target_arch = "loongarch64")]
    const _: () = {
        use crate::distance::half_c_objects::{LASX, LSX, same_features};
        assert!(same_features(COSINE_F16_LASX.features(), LASX));
        assert!(same_features(COSINE_BF16_LASX.features(), LASX));
        assert!(same_features(COSINE_F16_LSX.features(), LSX));
        assert!(same_features(COSINE_BF16_LSX.features(), LSX));
    };
}

impl Cosine for f32 {
    /// The kernel computes the norm of `x` itself, in f64: it need not be an
    /// f32 value.
    #[inline]
    fn cosine(x: &[Self], other: &[Self]) -> f32 {
        assert_equal_lengths(x.len(), other.len());
        COSINE_F32_PAIR.get().call(x, other)
    }

    #[inline]
    fn cosine_fast(x: &[Self], x_norm: Self, other: &[Self]) -> f32 {
        assert_equal_lengths(x.len(), other.len());
        COSINE_F32.get().call(x, x_norm, other)
    }

    #[inline]
    fn cosine_with_norms(x: &[Self], x_norm: Self, y_norm: Self, y: &[Self]) -> Self {
        assert_equal_lengths(x.len(), y.len());
        COSINE_WITH_NORMS_F32.get().call(x, x_norm, y_norm, y)
    }

    fn cosine_batch_into(x: &[Self], batch: &[Self], dimension: usize, output: &mut [f32]) {
        assert_batch_output_layout(x.len(), batch.len(), dimension, output.len());
        COSINE_F32_BATCH.get().call(x, batch, dimension, output);
    }
}

impl Cosine for f64 {
    #[inline]
    fn cosine_fast(x: &[Self], x_norm: f32, y: &[Self]) -> f32 {
        assert_equal_lengths(x.len(), y.len());
        COSINE_F64.get().call(x, x_norm, y)
    }
}

lance_core::dispatch! {
    static COSINE_F32: Dispatch<CosineKernel<f32>> = {
        #[cfg(target_arch = "x86_64")]
        {
            &[
                x86::COSINE_F32_AVX512,
                x86::COSINE_F32_AVX_FMA,
                x86::COSINE_F32_AVX,
                COSINE_F32_PORTABLE,
            ]
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            &[COSINE_F32_PORTABLE]
        }
    };
}

lance_core::dispatch! {
    static COSINE_F32_PAIR: Dispatch<PairKernel> = {
        #[cfg(target_arch = "x86_64")]
        {
            &[
                x86::COSINE_F32_PAIR_AVX512,
                x86::COSINE_F32_PAIR_AVX_FMA,
                x86::COSINE_F32_PAIR_AVX,
                COSINE_F32_PAIR_PORTABLE,
            ]
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            &[COSINE_F32_PAIR_PORTABLE]
        }
    };
}

lance_core::dispatch! {
    static COSINE_WITH_NORMS_F32: Dispatch<CosineWithNormsKernel> = {
        #[cfg(target_arch = "x86_64")]
        {
            &[
                x86::COSINE_WITH_NORMS_F32_AVX512,
                x86::COSINE_WITH_NORMS_F32_AVX_FMA,
                x86::COSINE_WITH_NORMS_F32_AVX,
                COSINE_WITH_NORMS_F32_PORTABLE,
            ]
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            &[COSINE_WITH_NORMS_F32_PORTABLE]
        }
    };
}

lance_core::dispatch! {
    /// Unlike `L2_F32_BATCH`, this has an AVX-512 candidate, so Haswell builds
    /// select at runtime, at one indirect call per batch. Every candidate runs
    /// the same f64 body and returns the same bits; AVX-512 hosts run it on
    /// 512-bit vectors.
    static COSINE_F32_BATCH: Dispatch<BatchKernel> = {
        #[cfg(target_arch = "x86_64")]
        {
            &[
                x86::COSINE_F32_BATCH_AVX512,
                x86::COSINE_F32_BATCH_AVX_FMA,
                x86::COSINE_F32_BATCH_AVX,
                COSINE_F32_BATCH_PORTABLE,
            ]
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            &[COSINE_F32_BATCH_PORTABLE]
        }
    };
}

lance_core::dispatch! {
    static COSINE_F64: Dispatch<CosineKernel<f64>> = {
        #[cfg(target_arch = "x86_64")]
        {
            &[
                x86::COSINE_F64_AVX512,
                x86::COSINE_F64_AVX_FMA,
                x86::COSINE_F64_AVX,
                COSINE_F64_PORTABLE,
            ]
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            &[COSINE_F64_PORTABLE]
        }
    };
}

/// Whether the portable kernels fuse their f64 multiply-adds: only where the
/// baseline has fused multiply-add, since `f64::mul_add` is a libm call
/// without it.
const PORTABLE_FMA: bool = cfg!(any(
    target_arch = "aarch64",
    target_arch = "loongarch64",
    target_feature = "fma"
));

/// `x·x`, `x·y` and `y·y`, accumulated in f64, where no product or partial sum
/// of f32 values can overflow or underflow. Every f32 candidate shares this
/// body and returns the same bits: a candidate's target features only change
/// how the compiler vectorizes it, and since the product of two f32 values is
/// exact in f64, `FMA` only decides whether each multiply-add is one
/// instruction. The compiler drops the sums a caller ignores.
#[inline(always)]
fn cosine_sums_f64<const FMA: bool>(x: &[f32], y: &[f32]) -> [f64; 3] {
    const LANES: usize = 8;
    let madd = |a: f64, b: f64, c: f64| if FMA { a.mul_add(b, c) } else { a * b + c };
    let (x_chunks, x_tail) = x.as_chunks::<LANES>();
    let (y_chunks, y_tail) = y.as_chunks::<LANES>();
    let mut xx = [0.0_f64; LANES];
    let mut xy = [0.0_f64; LANES];
    let mut yy = [0.0_f64; LANES];
    for (x, y) in x_chunks.iter().zip(y_chunks) {
        for i in 0..LANES {
            let (a, b) = (f64::from(x[i]), f64::from(y[i]));
            xx[i] = madd(a, a, xx[i]);
            xy[i] = madd(a, b, xy[i]);
            yy[i] = madd(b, b, yy[i]);
        }
    }
    let (mut xx_tail, mut xy_tail, mut yy_tail) = (0.0, 0.0, 0.0);
    for (&a, &b) in x_tail.iter().zip(y_tail) {
        let (a, b) = (f64::from(a), f64::from(b));
        xx_tail = madd(a, a, xx_tail);
        xy_tail = madd(a, b, xy_tail);
        yy_tail = madd(b, b, yy_tail);
    }
    let sum = |v: [f64; LANES]| ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
    [sum(xx) + xx_tail, sum(xy) + xy_tail, sum(yy) + yy_tail]
}

/// Cosine distance from `x`, whose L2 norm is `x_norm`, to `y`, in f64.
#[inline(always)]
fn cosine_f32_wide<const FMA: bool>(x: &[f32], x_norm: f64, y: &[f32]) -> f32 {
    let [_, xy, yy] = cosine_sums_f64::<FMA>(x, y);
    (1.0 - xy / x_norm / yy.sqrt()) as f32
}

/// Cosine distance between `x` and `y`, with both norms in f64.
#[inline(always)]
fn cosine_pair_f32_wide<const FMA: bool>(x: &[f32], y: &[f32]) -> f32 {
    let [xx, xy, yy] = cosine_sums_f64::<FMA>(x, y);
    (1.0 - xy / xx.sqrt() / yy.sqrt()) as f32
}

/// Whether `norm` is accurate if it came from an f32 sum of `dimension`
/// squares, as `norm_l2` computes it: the sum neither overflowed nor lost
/// precision to subnormal squares. The f32 kernels use a norm passed in only
/// when this holds, and otherwise compute it from the vector.
#[inline]
fn f32_norm_in_range(norm: f32, dimension: usize) -> bool {
    (f32::MIN_POSITIVE * dimension as f32..f32::INFINITY).contains(&(norm * norm))
}

/// [`Cosine::cosine_fast`] for f32, in f64.
#[inline(always)]
fn cosine_fast_f32_wide<const FMA: bool>(x: &[f32], x_norm: f32, y: &[f32]) -> f32 {
    if f32_norm_in_range(x_norm, x.len()) {
        cosine_f32_wide::<FMA>(x, x_norm.into(), y)
    } else {
        cosine_pair_f32_wide::<FMA>(x, y)
    }
}

/// [`Cosine::cosine_with_norms`] for f32, in f64.
#[inline(always)]
fn cosine_with_norms_f32_wide<const FMA: bool>(
    x: &[f32],
    x_norm: f32,
    y_norm: f32,
    y: &[f32],
) -> f32 {
    if f32_norm_in_range(x_norm, x.len()) && f32_norm_in_range(y_norm, y.len()) {
        let [_, xy, _] = cosine_sums_f64::<FMA>(x, y);
        (1.0 - xy / f64::from(x_norm) / f64::from(y_norm)) as f32
    } else {
        cosine_pair_f32_wide::<FMA>(x, y)
    }
}

/// [`Cosine::cosine_batch_into`] for f32, in f64. Fixed-size keys at the PQ
/// widths let the compiler unroll each vector and keep the key in registers.
#[inline(always)]
fn cosine_batch_f32_wide<const FMA: bool>(
    x: &[f32],
    batch: &[f32],
    dimension: usize,
    output: &mut [f32],
) {
    let x_norm = norm_l2_impl::<f32, f64, 16>(x);
    if dimension == 8
        && let Ok(key) = <&[f32; 8]>::try_from(x)
    {
        for (distance, y) in output.iter_mut().zip(batch.as_chunks::<8>().0) {
            *distance = cosine_f32_wide::<FMA>(key, x_norm, y);
        }
    } else if dimension == 16
        && let Ok(key) = <&[f32; 16]>::try_from(x)
    {
        for (distance, y) in output.iter_mut().zip(batch.as_chunks::<16>().0) {
            *distance = cosine_f32_wide::<FMA>(key, x_norm, y);
        }
    } else {
        for (distance, y) in output.iter_mut().zip(batch.chunks_exact(dimension)) {
            *distance = cosine_f32_wide::<FMA>(x, x_norm, y);
        }
    }
}

lance_core::target_kernel! {
    #[target_features()]
    #[candidate(COSINE_F32_PORTABLE: CosineKernel<f32>)]
    #[inline]
    fn cosine_f32_portable(x: &[f32], x_norm: f32, y: &[f32]) -> f32 {
        cosine_fast_f32_wide::<PORTABLE_FMA>(x, x_norm, y)
    }
}

lance_core::target_kernel! {
    #[target_features()]
    #[candidate(COSINE_F32_PAIR_PORTABLE: PairKernel)]
    #[inline]
    fn cosine_pair_f32_portable(x: &[f32], y: &[f32]) -> f32 {
        cosine_pair_f32_wide::<PORTABLE_FMA>(x, y)
    }
}

lance_core::target_kernel! {
    #[target_features()]
    #[candidate(COSINE_WITH_NORMS_F32_PORTABLE: CosineWithNormsKernel)]
    #[inline]
    fn cosine_with_norms_f32_portable(x: &[f32], x_norm: f32, y_norm: f32, y: &[f32]) -> f32 {
        cosine_with_norms_f32_wide::<PORTABLE_FMA>(x, x_norm, y_norm, y)
    }
}

lance_core::target_kernel! {
    #[target_features()]
    #[candidate(COSINE_F32_BATCH_PORTABLE: BatchKernel)]
    fn cosine_batch_f32_portable(x: &[f32], batch: &[f32], dimension: usize, output: &mut [f32]) {
        cosine_batch_f32_wide::<PORTABLE_FMA>(x, batch, dimension, output);
    }
}

#[cfg(target_arch = "x86_64")]
lance_core::target_kernel! {
    #[target_features()]
    #[candidate(COSINE_F64_PORTABLE: CosineKernel<f64>)]
    #[inline]
    fn cosine_f64_portable(x: &[f64], x_norm: f32, y: &[f64]) -> f32 {
        cosine_scalar(x, x_norm, y)
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::{
        CosineKernel, CosineWithNormsKernel, PairKernel, cosine_batch_f32_wide,
        cosine_fast_f32_wide, cosine_pair_f32_wide, cosine_with_norms_f32_wide,
    };
    use crate::distance::BatchKernel;
    use crate::simd::f64::{f64x4, f64x8};
    use crate::simd::x86::hsum256_pd;
    use crate::simd::{FloatSimd, SIMD};

    lance_core::target_kernel! {
        #[target_features("avx512f")]
        #[candidate(pub(super) COSINE_F32_BATCH_AVX512: BatchKernel)]
        pub(super) fn cosine_batch_f32_avx512(
            x: &[f32],
            batch: &[f32],
            dimension: usize,
            output: &mut [f32],
        ) {
            cosine_batch_f32_wide::<true>(x, batch, dimension, output);
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx", "fma")]
        #[candidate(pub(super) COSINE_F32_BATCH_AVX_FMA: BatchKernel)]
        pub(super) fn cosine_batch_f32_avx_fma(
            x: &[f32],
            batch: &[f32],
            dimension: usize,
            output: &mut [f32],
        ) {
            cosine_batch_f32_wide::<true>(x, batch, dimension, output);
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx")]
        #[candidate(pub(super) COSINE_F32_BATCH_AVX: BatchKernel)]
        pub(super) fn cosine_batch_f32_avx(
            x: &[f32],
            batch: &[f32],
            dimension: usize,
            output: &mut [f32],
        ) {
            cosine_batch_f32_wide::<false>(x, batch, dimension, output);
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx512f")]
        #[candidate(pub(super) COSINE_F32_AVX512: CosineKernel<f32>)]
        #[inline]
        pub(super) fn cosine_f32_avx512(x: &[f32], x_norm: f32, other: &[f32]) -> f32 {
            cosine_fast_f32_wide::<true>(x, x_norm, other)
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx", "fma")]
        #[candidate(pub(super) COSINE_F32_AVX_FMA: CosineKernel<f32>)]
        #[inline]
        pub(super) fn cosine_f32_avx_fma(x: &[f32], x_norm: f32, other: &[f32]) -> f32 {
            cosine_fast_f32_wide::<true>(x, x_norm, other)
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx")]
        #[candidate(pub(super) COSINE_F32_AVX: CosineKernel<f32>)]
        #[inline]
        pub(super) fn cosine_f32_avx(x: &[f32], x_norm: f32, other: &[f32]) -> f32 {
            cosine_fast_f32_wide::<false>(x, x_norm, other)
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx512f")]
        #[candidate(pub(super) COSINE_F32_PAIR_AVX512: PairKernel)]
        #[inline]
        pub(super) fn cosine_pair_f32_avx512(x: &[f32], y: &[f32]) -> f32 {
            cosine_pair_f32_wide::<true>(x, y)
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx", "fma")]
        #[candidate(pub(super) COSINE_F32_PAIR_AVX_FMA: PairKernel)]
        #[inline]
        pub(super) fn cosine_pair_f32_avx_fma(x: &[f32], y: &[f32]) -> f32 {
            cosine_pair_f32_wide::<true>(x, y)
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx")]
        #[candidate(pub(super) COSINE_F32_PAIR_AVX: PairKernel)]
        #[inline]
        pub(super) fn cosine_pair_f32_avx(x: &[f32], y: &[f32]) -> f32 {
            cosine_pair_f32_wide::<false>(x, y)
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx512f")]
        #[candidate(pub(super) COSINE_WITH_NORMS_F32_AVX512: CosineWithNormsKernel)]
        #[inline]
        pub(super) fn cosine_with_norms_f32_avx512(
            x: &[f32],
            x_norm: f32,
            y_norm: f32,
            y: &[f32],
        ) -> f32 {
            cosine_with_norms_f32_wide::<true>(x, x_norm, y_norm, y)
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx", "fma")]
        #[candidate(pub(super) COSINE_WITH_NORMS_F32_AVX_FMA: CosineWithNormsKernel)]
        #[inline]
        pub(super) fn cosine_with_norms_f32_avx_fma(
            x: &[f32],
            x_norm: f32,
            y_norm: f32,
            y: &[f32],
        ) -> f32 {
            cosine_with_norms_f32_wide::<true>(x, x_norm, y_norm, y)
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx")]
        #[candidate(pub(super) COSINE_WITH_NORMS_F32_AVX: CosineWithNormsKernel)]
        #[inline]
        pub(super) fn cosine_with_norms_f32_avx(
            x: &[f32],
            x_norm: f32,
            y_norm: f32,
            y: &[f32],
        ) -> f32 {
            cosine_with_norms_f32_wide::<false>(x, x_norm, y_norm, y)
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx512f")]
        #[candidate(pub(super) COSINE_F64_AVX512: CosineKernel<f64>)]
        /// 8-wide `__m512d` xy/yy with `vfmadd231pd` per iteration.
        #[inline]
        pub(super) fn cosine_f64_avx512(x: &[f64], x_norm: f32, y: &[f64]) -> f32 {
            // Callers pass equal lengths; `min` keeps the unchecked loads in
            // bounds for any input, as a safe function must.
            let dim = x.len().min(y.len());
            let unrolled_len = dim / 8 * 8;

            let mut acc_xy = _mm512_setzero_pd();
            let mut acc_yy = _mm512_setzero_pd();
            for i in (0..unrolled_len).step_by(8) {
                // SAFETY: `i + 8 <= dim`, which is in bounds for both slices.
                let (xv, yv) = unsafe {
                    (
                        _mm512_loadu_pd(x.as_ptr().add(i)),
                        _mm512_loadu_pd(y.as_ptr().add(i)),
                    )
                };
                acc_xy = _mm512_fmadd_pd(xv, yv, acc_xy);
                acc_yy = _mm512_fmadd_pd(yv, yv, acc_yy);
            }

            let mut xy = _mm512_reduce_add_pd(acc_xy);
            let mut yy = _mm512_reduce_add_pd(acc_yy);
            for i in unrolled_len..dim {
                xy += x[i] * y[i];
                yy += y[i] * y[i];
            }

            let y_norm_sq = yy as f32;
            let xy_f32 = xy as f32;
            1.0 - xy_f32 / x_norm / y_norm_sq.sqrt()
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx", "fma")]
        #[candidate(pub(super) COSINE_F64_AVX_FMA: CosineKernel<f64>)]
        /// Covers both AVX+FMA and AVX2 hosts (body uses no AVX2-specific intrinsics).
        #[inline]
        pub(super) fn cosine_f64_avx_fma(x: &[f64], x_norm: f32, y: &[f64]) -> f32 {
            // See `cosine_f64_avx512` for `min`.
            let dim = x.len().min(y.len());
            let unrolled_len = dim / 8 * 8;
            let mut y_norm8 = f64x8::zeros();
            let mut xy8 = f64x8::zeros();
            for i in (0..unrolled_len).step_by(8) {
                // SAFETY: `i + 8 <= dim`, which is in bounds for both slices.
                let (xv, yv) = unsafe {
                    (
                        f64x8::load_unaligned(x.as_ptr().add(i)),
                        f64x8::load_unaligned(y.as_ptr().add(i)),
                    )
                };
                xy8.multiply_add(xv, yv);
                y_norm8.multiply_add(yv, yv);
            }
            let aligned_len = dim / 4 * 4;
            let mut y_norm4 = f64x4::zeros();
            let mut xy4 = f64x4::zeros();
            for i in (unrolled_len..aligned_len).step_by(4) {
                // SAFETY: `i + 4 <= dim`, which is in bounds for both slices.
                let (xv, yv) = unsafe {
                    (
                        f64x4::load_unaligned(x.as_ptr().add(i)),
                        f64x4::load_unaligned(y.as_ptr().add(i)),
                    )
                };
                xy4.multiply_add(xv, yv);
                y_norm4.multiply_add(yv, yv);
            }
            let tail_y_norm: f64 = y[aligned_len..].iter().map(|&v| v * v).sum();
            let tail_xy: f64 = x[aligned_len..]
                .iter()
                .zip(y[aligned_len..].iter())
                .map(|(&a, &b)| a * b)
                .sum();

            let y_norm_sq = (y_norm8.reduce_sum() + y_norm4.reduce_sum() + tail_y_norm) as f32;
            let xy = (xy8.reduce_sum() + xy4.reduce_sum() + tail_xy) as f32;
            1.0 - xy / x_norm / y_norm_sq.sqrt()
        }
    }

    lance_core::target_kernel! {
        #[target_features("avx")]
        #[candidate(pub(super) COSINE_F64_AVX: CosineKernel<f64>)]
        /// AVX-only path (no FMA) for Sandy/Ivy Bridge.
        #[inline]
        pub(super) fn cosine_f64_avx(x: &[f64], x_norm: f32, y: &[f64]) -> f32 {
            // See `cosine_f64_avx512` for `min`.
            let dim = x.len().min(y.len());
            let aligned_len = dim / 4 * 4;

            let mut acc_xy = _mm256_setzero_pd();
            let mut acc_yy = _mm256_setzero_pd();
            for i in (0..aligned_len).step_by(4) {
                // SAFETY: `i + 4 <= dim`, which is in bounds for both slices.
                let (xv, yv) = unsafe {
                    (
                        _mm256_loadu_pd(x.as_ptr().add(i)),
                        _mm256_loadu_pd(y.as_ptr().add(i)),
                    )
                };
                acc_xy = _mm256_add_pd(acc_xy, _mm256_mul_pd(xv, yv));
                acc_yy = _mm256_add_pd(acc_yy, _mm256_mul_pd(yv, yv));
            }

            // SAFETY: AVX is enabled for this function.
            let (xy_main, yy_main) = unsafe { (hsum256_pd(acc_xy), hsum256_pd(acc_yy)) };

            let tail_y_norm: f64 = y[aligned_len..].iter().map(|&v| v * v).sum();
            let tail_xy: f64 = x[aligned_len..]
                .iter()
                .zip(y[aligned_len..].iter())
                .map(|(&a, &b)| a * b)
                .sum();

            let y_norm_sq = (yy_main + tail_y_norm) as f32;
            let xy = (xy_main + tail_xy) as f32;
            1.0 - xy / x_norm / y_norm_sq.sqrt()
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
lance_core::target_kernel! {
    #[target_features()]
    #[candidate(COSINE_F64_PORTABLE: CosineKernel<f64>)]
    /// `simd::f64` is backed by the baseline NEON or LSX on these architectures.
    #[inline]
    fn cosine_f64_portable(x: &[f64], x_norm: f32, y: &[f64]) -> f32 {
        use crate::simd::f64::{f64x4, f64x8};

        // Callers pass equal lengths; `min` keeps the unchecked loads in
        // bounds for any input, as a safe function must.
        let dim = x.len().min(y.len());
        let unrolled_len = dim / 8 * 8;
        let mut y_norm8 = f64x8::zeros();
        let mut xy8 = f64x8::zeros();
        for i in (0..unrolled_len).step_by(8) {
            // SAFETY: `i + 8 <= dim`, which is in bounds for both slices.
            let (xv, yv) = unsafe {
                (
                    f64x8::load_unaligned(x.as_ptr().add(i)),
                    f64x8::load_unaligned(y.as_ptr().add(i)),
                )
            };
            xy8.multiply_add(xv, yv);
            y_norm8.multiply_add(yv, yv);
        }
        let aligned_len = dim / 4 * 4;
        let mut y_norm4 = f64x4::zeros();
        let mut xy4 = f64x4::zeros();
        for i in (unrolled_len..aligned_len).step_by(4) {
            // SAFETY: `i + 4 <= dim`, which is in bounds for both slices.
            let (xv, yv) = unsafe {
                (
                    f64x4::load_unaligned(x.as_ptr().add(i)),
                    f64x4::load_unaligned(y.as_ptr().add(i)),
                )
            };
            xy4.multiply_add(xv, yv);
            y_norm4.multiply_add(yv, yv);
        }
        let tail_y_norm: f64 = y[aligned_len..].iter().map(|&v| v * v).sum();
        let tail_xy: f64 = x[aligned_len..]
            .iter()
            .zip(y[aligned_len..].iter())
            .map(|(&a, &b)| a * b)
            .sum();

        let y_norm_sq = (y_norm8.reduce_sum() + y_norm4.reduce_sum() + tail_y_norm) as f32;
        let xy = (xy8.reduce_sum() + xy4.reduce_sum() + tail_xy) as f32;
        1.0 - xy / x_norm / y_norm_sq.sqrt()
    }
}

/// Fallback non-SIMD implementation
#[inline]
fn cosine_scalar<T: Dot>(x: &[T], x_norm: f32, y: &[T]) -> f32 {
    let y_sq = dot(y, y);
    let xy = dot(x, y);
    // 1 - xy / (sqrt(x_sq) * sqrt(y_sq))
    1.0 - xy / (x_norm * y_sq.sqrt())
}

#[inline]
pub(crate) fn cosine_scalar_fast<T: Dot>(x: &[T], x_norm: f32, y: &[T], y_norm: f32) -> f32 {
    let xy = dot(x, y);
    // 1 - xy / (sqrt(x_sq) * sqrt(y_sq))
    // use f64 for overflow protection.
    1.0 - (xy / (x_norm * y_norm))
}

/// Cosine distance function between two vectors.
pub fn cosine_distance<T: Cosine>(from: &[T], to: &[T]) -> f32 {
    T::cosine(from, to)
}

/// Writes the [cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity)
/// from `from` to each `dimension`-sized vector in `batch` into the matching
/// element of `output`.
///
/// # Panics
///
/// Panics if `from.len() != dimension`, if `batch.len()` is not a multiple of
/// `dimension`, or if `output.len() != batch.len() / dimension`.
///
/// ```
/// # use lance_linalg::distance::cosine_distance_batch_into;
/// let mut distances = [0.0; 2];
/// cosine_distance_batch_into(&[1.0_f32, 0.0], &[2.0, 0.0, 0.0, 3.0], 2, &mut distances);
/// assert_eq!(distances, [0.0, 1.0]);
/// ```
#[inline]
pub fn cosine_distance_batch_into<T: Cosine>(
    from: &[T],
    batch: &[T],
    dimension: usize,
    output: &mut [f32],
) {
    T::cosine_batch_into(from, batch, dimension, output);
}

/// Compute cosine distances, propagating the null buffer of `to`.
///
/// See [`super::DistanceType::batch`] for supported types and input validation.
#[deprecated(note = "Use DistanceType::Cosine.batch")]
pub fn cosine_distance_arrow_batch(
    from: &dyn Array,
    to: &FixedSizeListArray,
) -> Result<Arc<Float32Array>> {
    super::DistanceType::Cosine.batch(from, to).map(Arc::new)
}

/// Portable scalar reference cosine over f64 inputs. Used by tests to compare
/// against every cosine kernel candidate. Computes
/// `1 - xy / (x_norm * y_norm_sq.sqrt())` in f64 then casts to f32, matching
/// the reduction order of the dispatched kernels.
#[cfg(test)]
fn cosine_fast_scalar(x: &[f64], x_norm: f32, y: &[f64]) -> f32 {
    let xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    let y_norm_sq: f64 = y.iter().map(|&v| v * v).sum();
    1.0 - (xy as f32) / x_norm / (y_norm_sq as f32).sqrt()
}

/// Portable scalar reference cosine when both norms are known. Mirrors
/// `Cosine::cosine_with_norms` for testing.
#[cfg(test)]
fn cosine_with_norms_scalar(x: &[f64], x_norm: f32, y_norm: f32, y: &[f64]) -> f32 {
    let xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    1.0 - (xy as f32) / x_norm / y_norm
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_utils::{
        arbitrary_bf16, arbitrary_f16, arbitrary_f32, arbitrary_f64, arbitrary_vector_pair,
        dimension_shard, run_vector_pair_proptest,
    };
    use approx::assert_relative_eq;
    use lance_core::utils::cpu::Dispatch;
    use num_traits::AsPrimitive;
    use proptest::prelude::*;

    fn cosine_dist_brute_force(x: &[f32], y: &[f32]) -> f32 {
        let xy = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi * yi)
            .sum::<f32>();
        let x_sq = x.iter().map(|&xi| xi * xi).sum::<f32>().sqrt();
        let y_sq = y.iter().map(|&yi| yi * yi).sum::<f32>().sqrt();
        1.0 - xy / x_sq / y_sq
    }

    #[test]
    fn half_cosine_rejects_mismatched_lengths_before_ffi() {
        let f16_x = [f16::from_f32(1.0)];
        let f16_y = [f16::from_f32(1.0), f16::from_f32(2.0)];
        assert!(
            std::panic::catch_unwind(|| <f16 as Cosine>::cosine_fast(&f16_x, 1.0, &f16_y)).is_err()
        );

        let bf16_x = [bf16::from_f32(1.0)];
        let bf16_y = [bf16::from_f32(1.0), bf16::from_f32(2.0)];
        assert!(
            std::panic::catch_unwind(|| <bf16 as Cosine>::cosine_fast(&bf16_x, 1.0, &bf16_y))
                .is_err()
        );
    }

    #[test]
    fn float_cosine_rejects_mismatched_lengths() {
        let f32_x = [1.0_f32];
        let f32_y = [1.0_f32, 2.0];
        assert!(
            std::panic::catch_unwind(|| <f32 as Cosine>::cosine_fast(&f32_x, 1.0, &f32_y)).is_err()
        );
        assert!(
            std::panic::catch_unwind(|| <f32 as Cosine>::cosine_with_norms(
                &f32_x, 1.0, 1.0, &f32_y
            ))
            .is_err()
        );

        let f64_x = [1.0_f64];
        let f64_y = [1.0_f64, 2.0];
        assert!(
            std::panic::catch_unwind(|| <f64 as Cosine>::cosine_fast(&f64_x, 1.0, &f64_y)).is_err()
        );
    }

    #[test]
    fn test_cosine() {
        let x: Float32Array = (1..9).map(|v| v as f32).collect();
        let y: Float32Array = (100..108).map(|v| v as f32).collect();
        let d = cosine_distance_batch_vec(x.values(), y.values(), 8);
        // from scipy.spatial.distance.cosine
        assert_relative_eq!(d[0], 1.0 - 0.900_957);

        let x = Float32Array::from_iter_values([3.0, 45.0, 7.0, 2.0, 5.0, 20.0, 13.0, 12.0]);
        let y = Float32Array::from_iter_values([2.0, 54.0, 13.0, 15.0, 22.0, 34.0, 50.0, 1.0]);
        let d = cosine_distance_batch_vec(x.values(), y.values(), 8);
        // from sklearn.metrics.pairwise import cosine_similarity
        assert_relative_eq!(d[0], 1.0 - 0.873_580_63);
    }

    #[test]
    fn test_cosine_large() {
        let total = 1024;
        let x = (0..total).map(|v| v as f32).collect::<Vec<_>>();
        let y = (1024..1024 + total).map(|v| v as f32).collect::<Vec<_>>();
        let d = cosine_distance_batch_vec(&x, &y, total);
        assert_relative_eq!(d[0], cosine_dist_brute_force(&x, &y));
    }

    #[test]
    fn test_cosine_not_aligned() {
        let x: Float32Array = vec![16_f32, 32_f32].into();
        let y: Float32Array = vec![1_f32, 2_f32, 4_f32, 8_f32].into();
        let d = cosine_distance_batch_vec(x.values(), y.values(), 2);
        assert_relative_eq!(d[0], 0.0);
        assert_relative_eq!(d[0], 0.0);
    }

    /// Reference implementation of cosine distance, plus error propagation.
    ///
    /// Pass `rel_err` to provide the allowed relative error in the dot product
    /// results. This function will then compute the expected absolute error.
    fn cosine_ref(x: &[f64], y: &[f64], rel_err: f64) -> (f32, f32) {
        let xy = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| xi * yi)
            .sum::<f64>();
        let x_sq = x.iter().map(|&xi| xi * xi).sum::<f64>().sqrt();
        let y_sq = y.iter().map(|&yi| yi * yi).sum::<f64>().sqrt();
        let expected = (1.0 - xy / x_sq / y_sq) as f32;

        let factor = 1.0 + rel_err;
        let low = (1.0 - (xy * factor) / (x_sq / factor) / (y_sq / factor)) as f32;
        let high = (1.0 - (xy / factor) / (x_sq * factor) / (y_sq * factor)) as f32;
        let low = (expected - low).abs();
        let high = (expected - high).abs();
        let error = low.max(high);

        (expected, error)
    }

    fn do_cosine_test<T: Cosine + AsPrimitive<f64>>(
        x: &[T],
        y: &[T],
    ) -> std::result::Result<(), TestCaseError> {
        let x_f64 = x.iter().map(|&v| v.as_()).collect::<Vec<_>>();
        let y_f64 = y.iter().map(|&v| v.as_()).collect::<Vec<_>>();

        let (expected, max_error) = cosine_ref(&x_f64, &y_f64, 1e-6);
        let result = T::cosine(x, y);

        prop_assert!(approx::relative_eq!(result, expected, epsilon = max_error));
        Ok(())
    }

    #[rstest::rstest]
    fn test_cosine_f32(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)] shard: usize,
    ) {
        run_vector_pair_proptest(arbitrary_f32, dimension_shard(shard), |x, y| {
            prop_assume!(norm_l2(&x) > 1e-10);
            prop_assume!(norm_l2(&y) > 1e-10);
            do_cosine_test(&x, &y)
        });
    }

    #[rstest::rstest]
    fn test_cosine_f64(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)] shard: usize,
    ) {
        run_vector_pair_proptest(arbitrary_f64, dimension_shard(shard), |x, y| {
            prop_assume!(norm_l2(&x) > 1e-20);
            prop_assume!(norm_l2(&y) > 1e-20);
            do_cosine_test(&x, &y)
        });
    }

    /// Checks one cosine distance against an f64 reference. The absolute
    /// `epsilon` covers inputs with extreme dynamic range, where a near-zero
    /// result differs between f32 and f64 accumulation by more than
    /// `max_relative` of itself.
    fn check_cosine(
        got: f32,
        want: f32,
        kernel: &str,
        dimension: usize,
    ) -> std::result::Result<(), TestCaseError> {
        prop_assert!(
            approx::relative_eq!(got, want, max_relative = 1e-3, epsilon = 1e-6),
            "{kernel} at dim {dimension}: {got} != {want}"
        );
        Ok(())
    }

    /// Every host-supported f32 kernel, single-pair and batch, must agree with
    /// an f64 reference. Prefixes of length 8 and 16 pin the batch kernels'
    /// PQ-width fast paths regardless of the generated dimension.
    #[rstest::rstest]
    fn test_cosine_f32_kernels_match_reference(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)] shard: usize,
    ) {
        run_vector_pair_proptest(arbitrary_f32, dimension_shard(shard), |x, y| {
            for dimension in [8, 16, x.len()].into_iter().filter(|&d| d <= x.len()) {
                let (x, y) = (&x[..dimension], &y[..dimension]);
                let (x_norm, y_norm) = (norm_l2(x), norm_l2(y));
                if x_norm <= 1e-10 || y_norm <= 1e-10 {
                    continue;
                }
                let x_f64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
                let y_f64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
                let want = cosine_fast_scalar(&x_f64, x_norm, &y_f64);
                let want_with_norms = cosine_with_norms_scalar(&x_f64, x_norm, y_norm, &y_f64);
                let want_self = cosine_fast_scalar(&x_f64, x_norm, &x_f64);

                for candidate in COSINE_F32.candidates() {
                    let Some(kernel) = candidate.kernel() else {
                        continue;
                    };
                    check_cosine(kernel.call(x, x_norm, y), want, candidate.name(), dimension)?;
                }

                // Every pair candidate shares one f64 body.
                let (want_pair, _) = cosine_ref(&x_f64, &y_f64, 0.0);
                let mut first_pair = None;
                for candidate in COSINE_F32_PAIR.candidates() {
                    let Some(kernel) = candidate.kernel() else {
                        continue;
                    };
                    let got = kernel.call(x, y);
                    check_cosine(got, want_pair, candidate.name(), dimension)?;
                    let (first, first_bits) =
                        *first_pair.get_or_insert((candidate.name(), got.to_bits()));
                    prop_assert_eq!(
                        got.to_bits(),
                        first_bits,
                        "{} and {} differ at dim {}",
                        candidate.name(),
                        first,
                        dimension
                    );
                }

                for candidate in COSINE_WITH_NORMS_F32.candidates() {
                    let Some(kernel) = candidate.kernel() else {
                        continue;
                    };
                    let got = kernel.call(x, x_norm, y_norm, y);
                    check_cosine(got, want_with_norms, candidate.name(), dimension)?;
                }

                let batch = [y, x].concat();
                let mut first_result = None;
                for candidate in COSINE_F32_BATCH.candidates() {
                    let Some(kernel) = candidate.kernel() else {
                        continue;
                    };
                    let mut got = [f32::NAN; 2];
                    kernel.call(x, &batch, dimension, &mut got);
                    check_cosine(got[0], want, candidate.name(), dimension)?;
                    check_cosine(got[1], want_self, candidate.name(), dimension)?;

                    // Every candidate shares one f64 body, so they return the
                    // same bits at every width.
                    let bits = got.map(f32::to_bits);
                    let (first, first_bits) = *first_result.get_or_insert((candidate.name(), bits));
                    prop_assert_eq!(
                        bits,
                        first_bits,
                        "{} and {} differ at dim {}",
                        candidate.name(),
                        first,
                        dimension
                    );
                }
            }
            Ok(())
        });
    }

    /// Every host-supported f64 kernel must agree with an f64 reference.
    #[rstest::rstest]
    fn test_cosine_f64_kernels_match_reference(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)] shard: usize,
    ) {
        run_vector_pair_proptest(arbitrary_f64, dimension_shard(shard), |x, y| {
            prop_assume!(norm_l2(&x) > 1e-20);
            prop_assume!(norm_l2(&y) > 1e-20);
            let x_norm = norm_l2(&x);
            let want = cosine_fast_scalar(&x, x_norm, &y);
            for candidate in COSINE_F64.candidates() {
                let Some(kernel) = candidate.kernel() else {
                    continue;
                };
                check_cosine(kernel.call(&x, x_norm, &y), want, candidate.name(), x.len())?;
            }
            Ok(())
        });
    }

    /// Vectors whose squared norm leaves the f32 range (#11269). In f32,
    /// `[1e20, 1e20, 0, ..]` squares to `inf`, `[1e-21, 1e-21, 0, ..]` to
    /// imprecise subnormals and `[1e-30, 1e-30, 0, ..]` to zero, and the norms
    /// of `[3e38, 3e38, 0, ..]` and `[1e-45, 1e-45, 0, ..]` are not f32 values
    /// at all, so an aligned vector came back at distance 1, NaN or `-inf`,
    /// and flat search ranked it last or dropped it. Each query is checked
    /// against `[scale, scale, 0, ..]` (distance 0) and `[1, 0, ..]`
    /// (distance `1 - 1/sqrt(2)`) through every f32 entry point and candidate.
    /// Where an entry point takes norms, they come from `norm_l2` and
    /// `norm_l2_fsl`, as the cached-norm callers compute them.
    #[rstest::rstest]
    fn test_cosine_f32_out_of_range_norms(
        #[values(1e20, 3e38, 1e-21, 1e-30, 1e-45)] scale: f32,
        #[values(8, 16, 40)] dimension: usize,
    ) {
        let vector = |head: &[f32]| {
            let mut v = vec![0.0_f32; dimension];
            v[..head.len()].copy_from_slice(head);
            v
        };
        let scaled = vector(&[scale, scale]);
        let batch = [scaled.as_slice(), &vector(&[1.0])].concat();
        let targets = FixedSizeListArray::try_new(
            Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Float32,
                true,
            )),
            dimension as i32,
            Arc::new(Float32Array::from(batch.clone())),
            None,
        )
        .unwrap();
        let target_norms = crate::distance::norm_l2_fsl(&targets).unwrap();
        let want = [0.0, 1.0 - std::f64::consts::FRAC_1_SQRT_2];

        for x in [&scaled, &vector(&[1.0, 1.0])] {
            let x_norm = norm_l2(x);
            let check = |got: [f32; 2], path: &str| {
                for (got, want) in got.into_iter().zip(want) {
                    assert!(
                        (f64::from(got) - want).abs() <= 1e-6,
                        "{path}: x = [{:e}, ..], scale {scale:e}, dim {dimension}: {got} != {want}",
                        x[0]
                    );
                }
            };
            let each_target = |distance: &dyn Fn(&[f32], f32) -> f32| {
                let mut got = [f32::NAN; 2];
                for ((got, y), &y_norm) in got
                    .iter_mut()
                    .zip(batch.chunks_exact(dimension))
                    .zip(target_norms.values())
                {
                    *got = distance(y, y_norm);
                }
                got
            };

            let got = crate::distance::DistanceType::Cosine
                .batch(&Float32Array::from(x.clone()), &targets)
                .unwrap();
            check([got.value(0), got.value(1)], "DistanceType::Cosine.batch");
            check(each_target(&|y, _| f32::cosine(x, y)), "Cosine::cosine");
            for candidate in COSINE_F32_PAIR.candidates() {
                let Some(kernel) = candidate.kernel() else {
                    continue;
                };
                check(each_target(&|y, _| kernel.call(x, y)), candidate.name());
            }
            for candidate in COSINE_F32_BATCH.candidates() {
                let Some(kernel) = candidate.kernel() else {
                    continue;
                };
                let mut got = [f32::NAN; 2];
                kernel.call(x, &batch, dimension, &mut got);
                check(got, candidate.name());
            }
            for candidate in COSINE_F32.candidates() {
                let Some(kernel) = candidate.kernel() else {
                    continue;
                };
                check(
                    each_target(&|y, _| kernel.call(x, x_norm, y)),
                    candidate.name(),
                );
            }
            for candidate in COSINE_WITH_NORMS_F32.candidates() {
                let Some(kernel) = candidate.kernel() else {
                    continue;
                };
                check(
                    each_target(&|y, y_norm| kernel.call(x, x_norm, y_norm, y)),
                    candidate.name(),
                );
            }
        }
    }

    /// The f32 entry points sum in f64 and round once, so an ordinary distance
    /// is correctly rounded. Every sum here is exact in f64.
    #[test]
    fn test_cosine_f32_rounds_once() {
        let (x, y) = ([1.0_f32, 2.0, 3.0], [4.0_f32, 5.0, 6.0]);
        let want = (1.0 - 32.0 / 14.0_f64.sqrt() / 77.0_f64.sqrt()) as f32;
        assert_eq!(f32::cosine(&x, &y), want);
        assert_eq!(cosine_distance_batch_vec(&x, &y, 3), [want]);
    }

    /// Every host-supported half-precision kernel must agree with an f64
    /// reference.
    fn check_half_cosine_kernels<T: Cosine + AsPrimitive<f64>>(
        dispatch: &Dispatch<CosineKernel<T>>,
        x: &[T],
        y: &[T],
    ) -> std::result::Result<(), TestCaseError> {
        let x_norm = norm_l2(x);
        prop_assume!(x_norm > 1e-6);
        prop_assume!(norm_l2(y) > 1e-6);
        let x_f64 = x.iter().map(|&v| v.as_()).collect::<Vec<_>>();
        let y_f64 = y.iter().map(|&v| v.as_()).collect::<Vec<_>>();
        let (want, max_error) = cosine_ref(&x_f64, &y_f64, 1e-6);
        for candidate in dispatch.candidates() {
            let Some(kernel) = candidate.kernel() else {
                continue;
            };
            let got = kernel.call(x, x_norm, y);
            prop_assert!(
                approx::relative_eq!(got, want, epsilon = max_error),
                "{} at dim {}: {got} != {want}",
                candidate.name(),
                x.len()
            );
            // The C candidates' safe wrappers clamp the length to the shorter
            // slice, so extra elements on either side must not change the
            // result. The scalar candidate is the one without features.
            if candidate.features().is_empty() {
                continue;
            }
            for (x, y) in [(x, &[y, x].concat()[..]), (&[x, y].concat()[..], y)] {
                let clamped = kernel.call(x, x_norm, y);
                prop_assert!(
                    clamped.to_bits() == got.to_bits(),
                    "{} at lengths {} and {}: {clamped} != {got}",
                    candidate.name(),
                    x.len(),
                    y.len()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_cosine_half_kernels_short_lengths() {
        run_vector_pair_proptest(arbitrary_f16, 0..4, |x, y| {
            check_half_cosine_kernels(&COSINE_F16, &x, &y)
        });
        run_vector_pair_proptest(arbitrary_bf16, 0..4, |x, y| {
            check_half_cosine_kernels(&COSINE_BF16, &x, &y)
        });
    }

    #[rstest::rstest]
    fn test_cosine_f16_kernels_match_reference(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)] shard: usize,
    ) {
        run_vector_pair_proptest(arbitrary_f16, dimension_shard(shard), |x, y| {
            check_half_cosine_kernels(&COSINE_F16, &x, &y)
        });
    }

    #[rstest::rstest]
    fn test_cosine_bf16_kernels_match_reference(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)] shard: usize,
    ) {
        run_vector_pair_proptest(arbitrary_bf16, dimension_shard(shard), |x, y| {
            check_half_cosine_kernels(&COSINE_BF16, &x, &y)
        });
    }

    proptest::proptest! {
        #[test]
        fn test_cosine_f16((x, y) in arbitrary_vector_pair(arbitrary_f16, 4..4048)) {
            // Cosine requires non-zero vectors
            prop_assume!(norm_l2(&x) > 1e-6);
            prop_assume!(norm_l2(&y) > 1e-6);
            do_cosine_test(&x, &y)?;
        }

        #[test]
        fn test_cosine_bf16((x, y) in arbitrary_vector_pair(arbitrary_bf16, 4..4048)){
            prop_assume!(norm_l2(&x) > 1e-6);
            prop_assume!(norm_l2(&y) > 1e-6);
            do_cosine_test(&x, &y)?;
        }
    }

    /// The f32 arrow path fills every distance with one batch kernel call; it
    /// must match the per-vector `cosine` and keep the targets' nulls.
    #[rstest::rstest]
    #[case::dim_8(8)]
    #[case::dim_40(40)]
    fn test_cosine_arrow_batch_f32_matches_per_vector(#[case] dimension: usize) {
        let num_vectors = 70;
        let x: Vec<f32> = (0..dimension)
            .map(|index| (index % 13) as f32 * 0.25 + 1.0)
            .collect();
        let batch: Vec<f32> = (0..dimension * num_vectors)
            .map(|index| (index % 11) as f32 * 0.5 - 2.0)
            .collect();
        let nulls = arrow_buffer::NullBuffer::from_iter((0..num_vectors).map(|i| i % 3 != 0));
        let to = FixedSizeListArray::try_new(
            Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Float32,
                true,
            )),
            dimension as i32,
            Arc::new(Float32Array::from(batch.clone())),
            Some(nulls.clone()),
        )
        .unwrap();

        let got = crate::distance::DistanceType::Cosine
            .batch(&Float32Array::from(x.clone()), &to)
            .unwrap();

        assert_eq!(got.nulls(), Some(&nulls));
        for (got, vector) in got.values().iter().zip(batch.chunks_exact(dimension)) {
            assert_relative_eq!(*got, f32::cosine(&x, vector), max_relative = 1e-5);
        }
    }

    fn cosine_distance_batch_vec<T: Cosine>(from: &[T], to: &[T], dimension: usize) -> Vec<f32> {
        let mut distances = vec![0.0; to.len() / dimension];
        cosine_distance_batch_into(from, to, dimension, &mut distances);
        distances
    }

    /// `cosine_batch_into` must agree with the per-vector `cosine`, both for
    /// the f32 batch kernel and for the per-vector default that f64 uses.
    #[rstest::rstest]
    #[case::dim_8(8)]
    #[case::dim_12(12)]
    #[case::dim_16(16)]
    #[case::dim_40(40)]
    #[case::dim_1024(1024)]
    fn test_cosine_batch_into_matches_per_vector(#[case] dimension: usize) {
        fn check<T: Cosine + num_traits::FromPrimitive + Copy>(dimension: usize) {
            let num_vectors = 130;
            let x: Vec<T> = (0..dimension)
                .map(|index| T::from_f32((index % 13) as f32 * 0.25 + 1.0).unwrap())
                .collect();
            let batch: Vec<T> = (0..dimension * num_vectors)
                .map(|index| T::from_f32((index % 11) as f32 * 0.5 - 2.0).unwrap())
                .collect();

            let mut got = vec![f32::NAN; num_vectors];
            T::cosine_batch_into(&x, &batch, dimension, &mut got);

            for (distance, vector) in got.iter().zip(batch.chunks_exact(dimension)) {
                let expected = T::cosine(&x, vector);
                assert_relative_eq!(*distance, expected, max_relative = 1e-5, epsilon = 1e-6);
            }
        }
        check::<f32>(dimension);
        check::<f64>(dimension);
    }

    #[rstest::rstest]
    #[case::short(1)]
    #[case::long(3)]
    fn test_cosine_batch_into_rejects_wrong_output_len(#[case] output_len: usize) {
        let batch = [1.0_f32; 4];
        let mut output = vec![0.0; output_len];
        let result = std::panic::catch_unwind(move || {
            cosine_distance_batch_into(&[1.0_f32, 2.0], &batch, 2, &mut output)
        });
        assert!(result.is_err());
    }
}
