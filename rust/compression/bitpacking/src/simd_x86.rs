// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Explicit x86_64 AVX-512 and AVX2 FastLanes bitpacking kernels with runtime dispatch.
//!
//! Each FastLanes virtual row spans 1024 bits (128 bytes) regardless of lane width
//! (`128 x u8`, `64 x u16`, `32 x u32`, or `16 x u64`). Representing a 128-byte row
//! explicitly as `[__m512i; 2]` (AVX-512) or `[__m256i; 4]` (AVX2) allows:
//! - Full 512-bit vector execution with 32 architectural `zmm` registers on AVX-512 hosts
//!   even when the workspace default target is `haswell`, eliminating register spills on
//!   wide multi-word bit-widths.
//! - Single-register mask reuse across all 16/32/64 rows of `u16`/`u32`/`u64` kernels.
//! - Native masked `epi16` bit shifts for `u8` FastLanes without `u8 <-> u16` widening.
//! - Runtime `AVX2` dispatch for downstream builds that do not set `-C target-cpu=haswell`.

use core::mem::MaybeUninit;
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum SimdLevel {
    Scalar = 1,
    Avx2 = 2,
    Avx512 = 3,
}

#[inline]
pub fn detect_level() -> SimdLevel {
    static LEVEL: AtomicU8 = AtomicU8::new(0);
    match LEVEL.load(Ordering::Relaxed) {
        3 => SimdLevel::Avx512,
        2 => SimdLevel::Avx2,
        1 => SimdLevel::Scalar,
        _ => init_level(&LEVEL),
    }
}

#[cold]
fn init_level(cell: &AtomicU8) -> SimdLevel {
    let level = if std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
    {
        SimdLevel::Avx512
    } else if std::arch::is_x86_feature_detected!("avx2") {
        SimdLevel::Avx2
    } else {
        SimdLevel::Scalar
    };
    cell.store(level as u8, Ordering::Relaxed);
    level
}

#[inline(always)]
const fn row_base(row: usize) -> usize {
    (crate::FL_ORDER[row / 8] * 16) + ((row % 8) * 128)
}

macro_rules! impl_x86_backend {
    (
        mod_name: $mod_name:ident,
        feature: $feat:literal,
        imm_ty: $imm_ty:ty,
        load_fn: $load_fn:ident,
        store_fn: $store_fn:ident,
        and_fn: $and_fn:ident,
        or_fn: $or_fn:ident,
        splat8: $splat8:ident,
        splat16: $splat16:ident,
        splat32: $splat32:ident,
        splat64: $splat64:ident,
        shl16: $shl16:ident,
        shr16: $shr16:ident,
        shl32: $shl32:ident,
        shr32: $shr32:ident,
        shl64: $shl64:ident,
        shr64: $shr64:ident,
    ) => {
        pub mod $mod_name {
            use super::{
                $and_fn, $load_fn, $or_fn, $shl16, $shl32, $shl64, $shr16, $shr32, $shr64,
                $splat8, $splat16, $splat32, $splat64, $store_fn, row_base,
            };
            use core::mem::MaybeUninit;

            // --- u8 FastLanes (128 lanes/row, masked epi16 shifts) ---

            #[target_feature(enable = $feat)]
            unsafe fn pack_8_full(input: *const u8, output: *mut MaybeUninit<u8>) {
                seq_macro::seq!(row in 0..8 {
                    // SAFETY: `input` and `output` each hold 1024 bytes (`8 * 128`).
                    unsafe {
                        let v = $load_fn(input, row * 128);
                        $store_fn(output, row * 128, v);
                    }
                });
            }

            #[target_feature(enable = $feat)]
            unsafe fn unpack_8_full(input: *const u8, output: *mut MaybeUninit<u8>) {
                seq_macro::seq!(row in 0..8 {
                    // SAFETY: `input` and `output` each hold 1024 bytes (`8 * 128`).
                    unsafe {
                        let v = $load_fn(input, row * 128);
                        $store_fn(output, row * 128, v);
                    }
                });
            }

            seq_macro::seq!(W in 1..8 {
                #[allow(unused_assignments)]
                #[target_feature(enable = $feat)]
                unsafe fn pack_8_~W(input: *const u8, output: *mut MaybeUninit<u8>) {
                    let mask_w = core::hint::black_box($splat8((1u8 << W) - 1));
                    // SAFETY: `input` points to 1024 bytes.
                    let mut tmp = $and_fn(unsafe { $load_fn(input, 0) }, mask_w);

                    seq_macro::seq!(row in 1..8 {
                        {
                            const CURR_WORD: usize = (row * W) / 8;
                            const NEXT_WORD: usize = ((row + 1) * W) / 8;
                            const SHIFT: $imm_ty = ((row * W) % 8) as $imm_ty;
                            const REM_BITS: usize = ((row + 1) * W) % 8;
                            const LOW_BITS: usize = (W as usize).saturating_sub(REM_BITS);
                            const LOW_SHIFT: $imm_ty = LOW_BITS as $imm_ty;

                            // SAFETY: `input` points to 1024 bytes and `row * 128 + 128 <= 1024`.
                            let raw = unsafe { $load_fn(input, row * 128) };
                            if SHIFT == 0 {
                                tmp = $and_fn(raw, mask_w);
                            } else if NEXT_WORD == CURR_WORD {
                                let src = $and_fn(raw, mask_w);
                                tmp = $or_fn(tmp, $shl16::<SHIFT>(src));
                            } else if REM_BITS == 0 {
                                let src = $and_fn(raw, mask_w);
                                let packed_word = $or_fn(tmp, $shl16::<SHIFT>(src));
                                // SAFETY: `output` holds `128 * W` bytes and `CURR_WORD < W`.
                                unsafe { $store_fn(output, 128 * CURR_WORD, packed_word) };
                            } else {
                                let low_src = $and_fn(raw, $splat8(((1u16 << LOW_BITS) - 1) as u8));
                                let packed_word = $or_fn(tmp, $shl16::<SHIFT>(low_src));
                                // SAFETY: `output` holds `128 * W` bytes and `CURR_WORD < W`.
                                unsafe { $store_fn(output, 128 * CURR_WORD, packed_word) };
                                tmp = $and_fn($shr16::<LOW_SHIFT>(raw), $splat8(((1u16 << REM_BITS) - 1) as u8));
                            }
                        }
                    });
                }

                #[allow(unused_assignments)]
                #[target_feature(enable = $feat)]
                unsafe fn unpack_8_~W(input: *const u8, output: *mut MaybeUninit<u8>) {
                    let mask_w = core::hint::black_box($splat8((1u8 << W) - 1));
                    // SAFETY: `input` holds `128 * W` bytes (`W >= 1`) and `output` holds 1024 bytes.
                    let mut src = unsafe { $load_fn(input, 0) };
                    // SAFETY: `output` holds 1024 bytes.
                    unsafe { $store_fn(output, 0, $and_fn(src, mask_w)) };

                    seq_macro::seq!(row in 1..8 {
                        {
                            const CURR_WORD: usize = (row * W) / 8;
                            const NEXT_WORD: usize = ((row + 1) * W) / 8;
                            const SHIFT: $imm_ty = ((row * W) % 8) as $imm_ty;
                            const REM_BITS: usize = ((row + 1) * W) % 8;
                            const CURRENT_BITS: usize = (W as usize).saturating_sub(REM_BITS);
                            const CUR_SHIFT: $imm_ty = CURRENT_BITS as $imm_ty;

                            if NEXT_WORD == CURR_WORD {
                                let elem = if SHIFT == 0 {
                                    $and_fn(src, mask_w)
                                } else {
                                    $and_fn($shr16::<SHIFT>(src), mask_w)
                                };
                                // SAFETY: `output` holds 1024 bytes and `row * 128 + 128 <= 1024`.
                                unsafe { $store_fn(output, row * 128, elem) };
                            } else if REM_BITS == 0 {
                                let elem = $and_fn($shr16::<SHIFT>(src), mask_w);
                                // SAFETY: `output` holds 1024 bytes and `row * 128 + 128 <= 1024`.
                                unsafe { $store_fn(output, row * 128, elem) };
                                if NEXT_WORD < W {
                                    // SAFETY: `input` holds `128 * W` bytes and `NEXT_WORD < W`.
                                    src = unsafe { $load_fn(input, 128 * NEXT_WORD) };
                                }
                            } else {
                                let low = $and_fn($shr16::<SHIFT>(src), $splat8(((1u16 << CURRENT_BITS) - 1) as u8));
                                // SAFETY: `input` holds `128 * W` bytes and `NEXT_WORD < W` when `REM_BITS > 0`.
                                src = unsafe { $load_fn(input, 128 * NEXT_WORD) };
                                let high = $shl16::<CUR_SHIFT>($and_fn(src, $splat8(((1u16 << REM_BITS) - 1) as u8)));
                                // SAFETY: `output` holds 1024 bytes and `row * 128 + 128 <= 1024`.
                                unsafe { $store_fn(output, row * 128, $or_fn(low, high)) };
                            }
                        }
                    });
                }
            });

            #[target_feature(enable = $feat)]
            pub unsafe fn pack_u8(
                width: usize,
                input: &[u8],
                output: &mut [MaybeUninit<u8>],
            ) {
                seq_macro::seq!(W in 1..8 {
                    match width {
                        0 => {}
                        #(
                            // SAFETY: Caller verified `input.len() == 1024`, `output.len() == 128 * W`,
                            // and target features match `$feat`.
                            W => unsafe { pack_8_~W(input.as_ptr(), output.as_mut_ptr()) },
                        )*
                        // SAFETY: Caller verified `input.len() == 1024` and `output.len() == 1024`.
                        8 => unsafe { pack_8_full(input.as_ptr(), output.as_mut_ptr()) },
                        _ => unreachable!("Unsupported width: {}", width),
                    }
                });
            }

            #[target_feature(enable = $feat)]
            pub unsafe fn unpack_u8(
                width: usize,
                input: &[u8],
                output: &mut [MaybeUninit<u8>],
            ) {
                seq_macro::seq!(W in 1..8 {
                    match width {
                        0 => output.fill(MaybeUninit::new(0)),
                        #(
                            // SAFETY: Caller verified `input.len() == 128 * W`, `output.len() == 1024`,
                            // and target features match `$feat`.
                            W => unsafe { unpack_8_~W(input.as_ptr(), output.as_mut_ptr()) },
                        )*
                        // SAFETY: Caller verified `input.len() == 1024` and `output.len() == 1024`.
                        8 => unsafe { unpack_8_full(input.as_ptr(), output.as_mut_ptr()) },
                        _ => unreachable!("Unsupported width: {}", width),
                    }
                });
            }

            // --- Shared generator for u16 / u32 / u64 FastLanes (native lane shifts, 1 mask reg) ---

            macro_rules! impl_word_type {
                (
                    $ty:ty,
                    $bits:literal,
                    $lanes:literal,
                    $splat:ident,
                    $shl:ident,
                    $shr:ident,
                    $pack_fn:ident,
                    $unpack_fn:ident,
                    $pack_full:ident,
                    $unpack_full:ident,
                    $pack_prefix:ident,
                    $unpack_prefix:ident
                ) => {
                    #[target_feature(enable = $feat)]
                    unsafe fn $pack_full(input: *const $ty, output: *mut MaybeUninit<$ty>) {
                        seq_macro::seq!(row in 0..$bits {
                            // SAFETY: `input` and `output` each hold 1024 elements, and both
                            // `row_base(row) + $lanes` and `$lanes * row + $lanes` are `<= 1024`.
                            unsafe {
                                let v = $load_fn(input, row_base(row));
                                $store_fn(output, $lanes * row, v);
                            }
                        });
                    }

                    #[target_feature(enable = $feat)]
                    unsafe fn $unpack_full(input: *const $ty, output: *mut MaybeUninit<$ty>) {
                        seq_macro::seq!(row in 0..$bits {
                            // SAFETY: `input` and `output` each hold 1024 elements, and both
                            // `$lanes * row + $lanes` and `row_base(row) + $lanes` are `<= 1024`.
                            unsafe {
                                let v = $load_fn(input, $lanes * row);
                                $store_fn(output, row_base(row), v);
                            }
                        });
                    }

                    seq_macro::seq!(W in 1..$bits {
                        #[allow(unused_assignments)]
                        #[target_feature(enable = $feat)]
                        unsafe fn $pack_prefix~W(input: *const $ty, output: *mut MaybeUninit<$ty>) {
                            let mask_w = core::hint::black_box($splat(((1 as $ty) << W) - 1));
                            // SAFETY: `input` holds 1024 elements.
                            let mut tmp = $and_fn(unsafe { $load_fn(input, 0) }, mask_w);

                            seq_macro::seq!(row in 1..$bits {
                                {
                                    const CURR_WORD: usize = (row * W) / $bits;
                                    const NEXT_WORD: usize = ((row + 1) * W) / $bits;
                                    const SHIFT: $imm_ty = ((row * W) % $bits) as $imm_ty;
                                    const REM_BITS: usize = ((row + 1) * W) % $bits;
                                    const LOW_SHIFT: $imm_ty =
                                        (W as usize).saturating_sub(REM_BITS) as $imm_ty;

                                    // SAFETY: `input` holds 1024 elements and `row_base(row) + $lanes <= 1024`.
                                    let raw = unsafe { $load_fn(input, row_base(row)) };
                                    if SHIFT == 0 {
                                        tmp = $and_fn(raw, mask_w);
                                    } else if NEXT_WORD == CURR_WORD {
                                        let src = $and_fn(raw, mask_w);
                                        tmp = $or_fn(tmp, $shl::<SHIFT>(src));
                                    } else if REM_BITS == 0 {
                                        // `SHIFT + W == $bits`: left shift naturally discards any bits `>= W`.
                                        let packed_word = $or_fn(tmp, $shl::<SHIFT>(raw));
                                        // SAFETY: `output` holds `$lanes * W` elements and `CURR_WORD < W`.
                                        unsafe { $store_fn(output, $lanes * CURR_WORD, packed_word) };
                                    } else {
                                        let src = $and_fn(raw, mask_w);
                                        let packed_word = $or_fn(tmp, $shl::<SHIFT>(src));
                                        // SAFETY: `output` holds `$lanes * W` elements and `CURR_WORD < W`.
                                        unsafe { $store_fn(output, $lanes * CURR_WORD, packed_word) };
                                        tmp = $shr::<LOW_SHIFT>(src);
                                    }
                                }
                            });
                        }

                        #[allow(unused_assignments)]
                        #[target_feature(enable = $feat)]
                        unsafe fn $unpack_prefix~W(input: *const $ty, output: *mut MaybeUninit<$ty>) {
                            let mask_w = core::hint::black_box($splat(((1 as $ty) << W) - 1));
                            // SAFETY: `input` holds `$lanes * W` elements (`W >= 1`) and `output` holds 1024.
                            let mut src = unsafe { $load_fn(input, 0) };
                            // SAFETY: `output` holds 1024 elements.
                            unsafe { $store_fn(output, 0, $and_fn(src, mask_w)) };

                            seq_macro::seq!(row in 1..$bits {
                                {
                                    const CURR_WORD: usize = (row * W) / $bits;
                                    const NEXT_WORD: usize = ((row + 1) * W) / $bits;
                                    const SHIFT: $imm_ty = ((row * W) % $bits) as $imm_ty;
                                    const REM_BITS: usize = ((row + 1) * W) % $bits;
                                    const CUR_SHIFT: $imm_ty =
                                        (W as usize).saturating_sub(REM_BITS) as $imm_ty;

                                    if NEXT_WORD == CURR_WORD {
                                        let elem = if SHIFT == 0 {
                                            $and_fn(src, mask_w)
                                        } else {
                                            $and_fn($shr::<SHIFT>(src), mask_w)
                                        };
                                        // SAFETY: `output` holds 1024 elements and `row_base(row) + $lanes <= 1024`.
                                        unsafe { $store_fn(output, row_base(row), elem) };
                                    } else if REM_BITS == 0 {
                                        // `SHIFT + W == $bits`: logical right shift leaves exact `W` low bits.
                                        let elem = $shr::<SHIFT>(src);
                                        // SAFETY: `output` holds 1024 elements and `row_base(row) + $lanes <= 1024`.
                                        unsafe { $store_fn(output, row_base(row), elem) };
                                        if NEXT_WORD < W {
                                            // SAFETY: `input` holds `$lanes * W` elements and `NEXT_WORD < W`.
                                            src = unsafe { $load_fn(input, $lanes * NEXT_WORD) };
                                        }
                                    } else {
                                        let low = $shr::<SHIFT>(src);
                                        // SAFETY: `input` holds `$lanes * W` elements and `NEXT_WORD < W` when `REM_BITS > 0`.
                                        src = unsafe { $load_fn(input, $lanes * NEXT_WORD) };
                                        let high = $and_fn($shl::<CUR_SHIFT>(src), mask_w);
                                        // SAFETY: `output` holds 1024 elements and `row_base(row) + $lanes <= 1024`.
                                        unsafe { $store_fn(output, row_base(row), $or_fn(low, high)) };
                                    }
                                }
                            });
                        }
                    });

                    #[target_feature(enable = $feat)]
                    pub unsafe fn $pack_fn(
                        width: usize,
                        input: &[$ty],
                        output: &mut [MaybeUninit<$ty>],
                    ) {
                        seq_macro::seq!(W in 1..$bits {
                            match width {
                                0 => {}
                                #(
                                    // SAFETY: Caller verified `input.len() == 1024`, `output.len() == $lanes * W`,
                                    // and target features match `$feat`.
                                    W => unsafe { $pack_prefix~W(input.as_ptr(), output.as_mut_ptr()) },
                                )*
                                // SAFETY: Caller verified `input.len() == 1024` and `output.len() == 1024`.
                                $bits => unsafe { $pack_full(input.as_ptr(), output.as_mut_ptr()) },
                                _ => unreachable!("Unsupported width: {}", width),
                            }
                        });
                    }

                    #[target_feature(enable = $feat)]
                    pub unsafe fn $unpack_fn(
                        width: usize,
                        input: &[$ty],
                        output: &mut [MaybeUninit<$ty>],
                    ) {
                        seq_macro::seq!(W in 1..$bits {
                            match width {
                                0 => output.fill(MaybeUninit::new(0)),
                                #(
                                    // SAFETY: Caller verified `input.len() == $lanes * W`, `output.len() == 1024`,
                                    // and target features match `$feat`.
                                    W => unsafe { $unpack_prefix~W(input.as_ptr(), output.as_mut_ptr()) },
                                )*
                                // SAFETY: Caller verified `input.len() == 1024` and `output.len() == 1024`.
                                $bits => unsafe { $unpack_full(input.as_ptr(), output.as_mut_ptr()) },
                                _ => unreachable!("Unsupported width: {}", width),
                            }
                        });
                    }
                };
            }

            impl_word_type!(
                u16,
                16,
                64,
                $splat16,
                $shl16,
                $shr16,
                pack_u16,
                unpack_u16,
                pack_16_full,
                unpack_16_full,
                pack_16_,
                unpack_16_
            );
            impl_word_type!(
                u32,
                32,
                32,
                $splat32,
                $shl32,
                $shr32,
                pack_u32,
                unpack_u32,
                pack_32_full,
                unpack_32_full,
                pack_32_,
                unpack_32_
            );
            impl_word_type!(
                u64,
                64,
                16,
                $splat64,
                $shl64,
                $shr64,
                pack_u64,
                unpack_u64,
                pack_64_full,
                unpack_64_full,
                pack_64_,
                unpack_64_
            );
        }
    };
}

// --- AVX-512 row primitives (`[__m512i; 2]` = 128 bytes) ---

use core::arch::x86_64::*;

type Row512 = [__m512i; 2];

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn load512<T>(base: *const T, elem_offset: usize) -> Row512 {
    // SAFETY: Caller guarantees `base.add(elem_offset)` has at least 128 readable bytes.
    unsafe {
        let p = base.add(elem_offset).cast::<u8>();
        [
            _mm512_loadu_si512(p.cast()),
            _mm512_loadu_si512(p.add(64).cast()),
        ]
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
unsafe fn store512<T>(base: *mut MaybeUninit<T>, elem_offset: usize, v: Row512) {
    // SAFETY: Caller guarantees `base.add(elem_offset)` has at least 128 writable bytes.
    unsafe {
        let p = base.add(elem_offset).cast::<u8>();
        _mm512_storeu_si512(p.cast(), v[0]);
        _mm512_storeu_si512(p.add(64).cast(), v[1]);
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn and512(a: Row512, b: Row512) -> Row512 {
    [_mm512_and_si512(a[0], b[0]), _mm512_and_si512(a[1], b[1])]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn or512(a: Row512, b: Row512) -> Row512 {
    [_mm512_or_si512(a[0], b[0]), _mm512_or_si512(a[1], b[1])]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn splat8_512(m: u8) -> Row512 {
    let v = _mm512_set1_epi8(m as i8);
    [v, v]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn splat16_512(m: u16) -> Row512 {
    let v = _mm512_set1_epi16(m as i16);
    [v, v]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn splat32_512(m: u32) -> Row512 {
    let v = _mm512_set1_epi32(m as i32);
    [v, v]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn splat64_512(m: u64) -> Row512 {
    let v = _mm512_set1_epi64(m as i64);
    [v, v]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn shl16_512<const IMM: u32>(a: Row512) -> Row512 {
    [
        _mm512_slli_epi16::<IMM>(a[0]),
        _mm512_slli_epi16::<IMM>(a[1]),
    ]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn shr16_512<const IMM: u32>(a: Row512) -> Row512 {
    [
        _mm512_srli_epi16::<IMM>(a[0]),
        _mm512_srli_epi16::<IMM>(a[1]),
    ]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn shl32_512<const IMM: u32>(a: Row512) -> Row512 {
    [
        _mm512_slli_epi32::<IMM>(a[0]),
        _mm512_slli_epi32::<IMM>(a[1]),
    ]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn shr32_512<const IMM: u32>(a: Row512) -> Row512 {
    [
        _mm512_srli_epi32::<IMM>(a[0]),
        _mm512_srli_epi32::<IMM>(a[1]),
    ]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn shl64_512<const IMM: u32>(a: Row512) -> Row512 {
    [
        _mm512_slli_epi64::<IMM>(a[0]),
        _mm512_slli_epi64::<IMM>(a[1]),
    ]
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
#[inline]
fn shr64_512<const IMM: u32>(a: Row512) -> Row512 {
    [
        _mm512_srli_epi64::<IMM>(a[0]),
        _mm512_srli_epi64::<IMM>(a[1]),
    ]
}

// --- AVX2 row primitives (`[__m256i; 4]` = 128 bytes) ---

type Row256 = [__m256i; 4];

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load256<T>(base: *const T, elem_offset: usize) -> Row256 {
    // SAFETY: Caller guarantees `base.add(elem_offset)` has at least 128 readable bytes.
    unsafe {
        let p = base.add(elem_offset).cast::<u8>();
        [
            _mm256_loadu_si256(p.cast()),
            _mm256_loadu_si256(p.add(32).cast()),
            _mm256_loadu_si256(p.add(64).cast()),
            _mm256_loadu_si256(p.add(96).cast()),
        ]
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store256<T>(base: *mut MaybeUninit<T>, elem_offset: usize, v: Row256) {
    // SAFETY: Caller guarantees `base.add(elem_offset)` has at least 128 writable bytes.
    unsafe {
        let p = base.add(elem_offset).cast::<u8>();
        _mm256_storeu_si256(p.cast(), v[0]);
        _mm256_storeu_si256(p.add(32).cast(), v[1]);
        _mm256_storeu_si256(p.add(64).cast(), v[2]);
        _mm256_storeu_si256(p.add(96).cast(), v[3]);
    }
}

#[target_feature(enable = "avx2")]
#[inline]
fn and256(a: Row256, b: Row256) -> Row256 {
    [
        _mm256_and_si256(a[0], b[0]),
        _mm256_and_si256(a[1], b[1]),
        _mm256_and_si256(a[2], b[2]),
        _mm256_and_si256(a[3], b[3]),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn or256(a: Row256, b: Row256) -> Row256 {
    [
        _mm256_or_si256(a[0], b[0]),
        _mm256_or_si256(a[1], b[1]),
        _mm256_or_si256(a[2], b[2]),
        _mm256_or_si256(a[3], b[3]),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn splat8_256(m: u8) -> Row256 {
    let v = _mm256_set1_epi8(m as i8);
    [v, v, v, v]
}

#[target_feature(enable = "avx2")]
#[inline]
fn splat16_256(m: u16) -> Row256 {
    let v = _mm256_set1_epi16(m as i16);
    [v, v, v, v]
}

#[target_feature(enable = "avx2")]
#[inline]
fn splat32_256(m: u32) -> Row256 {
    let v = _mm256_set1_epi32(m as i32);
    [v, v, v, v]
}

#[target_feature(enable = "avx2")]
#[inline]
fn splat64_256(m: u64) -> Row256 {
    let v = _mm256_set1_epi64x(m as i64);
    [v, v, v, v]
}

#[target_feature(enable = "avx2")]
#[inline]
fn shl16_256<const IMM: i32>(a: Row256) -> Row256 {
    [
        _mm256_slli_epi16::<IMM>(a[0]),
        _mm256_slli_epi16::<IMM>(a[1]),
        _mm256_slli_epi16::<IMM>(a[2]),
        _mm256_slli_epi16::<IMM>(a[3]),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn shr16_256<const IMM: i32>(a: Row256) -> Row256 {
    [
        _mm256_srli_epi16::<IMM>(a[0]),
        _mm256_srli_epi16::<IMM>(a[1]),
        _mm256_srli_epi16::<IMM>(a[2]),
        _mm256_srli_epi16::<IMM>(a[3]),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn shl32_256<const IMM: i32>(a: Row256) -> Row256 {
    [
        _mm256_slli_epi32::<IMM>(a[0]),
        _mm256_slli_epi32::<IMM>(a[1]),
        _mm256_slli_epi32::<IMM>(a[2]),
        _mm256_slli_epi32::<IMM>(a[3]),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn shr32_256<const IMM: i32>(a: Row256) -> Row256 {
    [
        _mm256_srli_epi32::<IMM>(a[0]),
        _mm256_srli_epi32::<IMM>(a[1]),
        _mm256_srli_epi32::<IMM>(a[2]),
        _mm256_srli_epi32::<IMM>(a[3]),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn shl64_256<const IMM: i32>(a: Row256) -> Row256 {
    [
        _mm256_slli_epi64::<IMM>(a[0]),
        _mm256_slli_epi64::<IMM>(a[1]),
        _mm256_slli_epi64::<IMM>(a[2]),
        _mm256_slli_epi64::<IMM>(a[3]),
    ]
}

#[target_feature(enable = "avx2")]
#[inline]
fn shr64_256<const IMM: i32>(a: Row256) -> Row256 {
    [
        _mm256_srli_epi64::<IMM>(a[0]),
        _mm256_srli_epi64::<IMM>(a[1]),
        _mm256_srli_epi64::<IMM>(a[2]),
        _mm256_srli_epi64::<IMM>(a[3]),
    ]
}

impl_x86_backend!(
    mod_name: avx512,
    feature: "avx512f,avx512bw,avx512vl",
    imm_ty: u32,
    load_fn: load512,
    store_fn: store512,
    and_fn: and512,
    or_fn: or512,
    splat8: splat8_512,
    splat16: splat16_512,
    splat32: splat32_512,
    splat64: splat64_512,
    shl16: shl16_512,
    shr16: shr16_512,
    shl32: shl32_512,
    shr32: shr32_512,
    shl64: shl64_512,
    shr64: shr64_512,
);

impl_x86_backend!(
    mod_name: avx2,
    feature: "avx2",
    imm_ty: i32,
    load_fn: load256,
    store_fn: store256,
    and_fn: and256,
    or_fn: or256,
    splat8: splat8_256,
    splat16: splat16_256,
    splat32: splat32_256,
    splat64: splat64_256,
    shl16: shl16_256,
    shr16: shr16_256,
    shl32: shl32_256,
    shr32: shr32_256,
    shl64: shl64_256,
    shr64: shr64_256,
);
