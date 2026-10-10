// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Poor-man's SIMD
//!
//! The difference between this implementation and [std::simd] is that
//! this implementation holds the SIMD register directly, thus it exposes more
//! optimization opportunity to use wide range of instructions available.
//!
//! Also, it gives us more control, for example, it is likely that we will have
//! f16/bf16 support before the standard library does.
//!
//! The API are close to [std::simd] to make migration easier in the future.

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

pub mod amx_fp16;
pub mod dist_table;
pub mod f32;
pub mod f64;
pub mod i32;
pub mod u8;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

use num_traits::{Float, Num};
use u8::u8x16;

/// Lance SIMD lib
///
pub trait SIMD<T: Num + Copy, const N: usize>:
    std::fmt::Debug
    + AddAssign<Self>
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Copy
    + Clone
    + Sized
    + for<'a> From<&'a [T]>
{
    const LANES: usize = N;

    /// Create a new instance with all lanes set to `val`.
    fn splat(val: T) -> Self;

    /// Create a new instance with all lanes set to zero.
    fn zeros() -> Self;

    /// Load aligned data from aligned memory.
    ///
    /// # Safety
    ///
    /// It crashes if the ptr is not aligned.
    unsafe fn load(ptr: *const T) -> Self;

    /// Load unaligned data from memory.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for reads of `N` elements of `T`.
    ///
    /// Most implementations also require a target feature that the build's
    /// baseline may not provide, and none of them check for it at runtime:
    ///
    /// - `f32x8`, `f32x16`, `f64x4`, `f64x8` and `i32x8` need AVX on x86_64 and
    ///   LASX on loongarch64;
    /// - `u8x16` needs nothing beyond the baseline on any target.
    ///
    /// Calling one without its feature is undefined behaviour rather than a
    /// wrong answer, so either build with the feature enabled, as
    /// `.cargo/config.toml` does for `x86_64-unknown-linux-gnu`, or check with
    /// `is_x86_feature_detected!` first. See #8872.
    unsafe fn load_unaligned(ptr: *const T) -> Self;

    /// Store the values to aligned memory.
    ///
    /// # Safety
    ///
    /// It crashes if the ptr is not aligned
    unsafe fn store(&self, ptr: *mut T);

    /// Store the values to unaligned memory.
    ///
    /// # Safety
    unsafe fn store_unaligned(&self, ptr: *mut T);

    /// Return the values as an array.
    fn as_array(&self) -> [T; N] {
        let mut arr = [T::zero(); N];
        unsafe {
            self.store_unaligned(arr.as_mut_ptr());
        }
        arr
    }

    /// Calculate the sum across this vector.
    fn reduce_sum(&self) -> T;

    /// Find the minimal value in the vector.
    fn reduce_min(&self) -> T;

    /// Return the minimal value of these two vectors.
    fn min(&self, rhs: &Self) -> Self;

    /// Find the index of value in the vector. If not found, return None.
    fn find(&self, val: T) -> Option<i32>;
}

pub trait FloatSimd<F: Float, const N: usize>: SIMD<F, N> {
    /// fused multiply-add
    ///
    /// c = a * b + c
    fn multiply_add(&mut self, a: Self, b: Self);
}

pub trait Shuffle {
    fn shuffle(&self, indices: u8x16) -> Self;
}
