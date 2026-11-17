// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// the first 32-bits of a FSST compressed file is the FSST magic number
const FSST_MAGIC: u64 = 0x46535354 << 32; // "FSST"
// when the code is FSST_ESC, the next byte should be interpreted as is
const FSST_ESC: u8 = 255;
// when building symbol table, we have a maximum of 512 symbols, so we can use 9 bits to represent the code
const FSST_CODE_BITS: u16 = 9;
// when building symbol table, we use the first 256 codes to represent the index itself, for example, code 0 represents byte 0
const FSST_CODE_BASE: u16 = 256;

// code 512, which we can never reach(maximum code is 511)
const FSST_CODE_MAX: u16 = 1 << FSST_CODE_BITS;
// all code bits set
const FSST_CODE_MASK: u16 = FSST_CODE_MAX - 1;
// we construct FSST symbol tables using a random sample of about 16KB (1<<14)
const FSST_SAMPLETARGET: usize = 1 << 14;
const FSST_SAMPLEMAXSZ: usize = 2 * FSST_SAMPLETARGET;

// if the input size is less than 32 KB, we mark the file header and copy the input to the output as is
pub const FSST_LEAST_INPUT_SIZE: usize = 32 * 1024;

// if the max length of the input strings are less than `FSST_LEAST_INPUT_MAX_LENGTH`, we shouldn't use FSST.
pub const FSST_LEAST_INPUT_MAX_LENGTH: u64 = 5;

// we only use the lower 32 bits in icl, so we can use 1 << 32 to represent a free slot in the hash table
const FSST_ICL_FREE: u64 = 1 << 32;
// in the icl field of a symbol, the symbol length is stored in 4 bits starting from the 28th bit
const CODE_LEN_SHIFT_IN_ICL: u64 = 28;
// in the icl field of a symbol, the symbol code is stored in the 12 bits starting from the 16th bit
const CODE_SHIFT_IN_ICL: u64 = 16;

const CODE_LEN_SHIFT_IN_CODE: u64 = 12;

const FSST_HASH_TAB_SIZE: usize = 1024;
const FSST_HASH_PRIME: u64 = 2971215073;
const FSST_SHIFT: usize = 15;

#[inline(always)]
fn fsst_hash(w: u64) -> u64 {
    let p = w.wrapping_mul(FSST_HASH_PRIME);
    p ^ (p >> FSST_SHIFT)
}

const MAX_SYMBOL_LENGTH: usize = 8;

pub const FSST_SYMBOL_TABLE_SIZE: usize = 8 + 256 * 8 + 256; // 8 bytes for the header, 256 symbols(8 bytes each), 256 bytes for lens

use arrow_array::OffsetSizeTrait;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::io;
use std::ptr;

#[inline(always)]
fn fsst_unaligned_load_unchecked(v: *const u8) -> u64 {
    // SAFETY: the caller must guarantee that `v` points to at least 8 readable bytes. All callers
    // uphold this:
    // - `compress_bulk` guards direct string loads with `in_curr + 8 <= end_curr`, and tail loads
    //   read from a 16-byte stack buffer at offset `< rem <= 7` (leaving `>= 9` readable bytes);
    // - `build_symbol_table` guards loads with `word.len() > 7 && curr < word.len() - 7`;
    // - `find_longest_symbol_from_char_slice` copies into a stack `[u8; 8]` before loading;
    // - `FsstDecompressor::try_new` reads symbols from a `symbol_table` buffer already validated
    //   to be `FSST_SYMBOL_TABLE_SIZE` bytes.
    unsafe { ptr::read_unaligned(v as *const u64) }
}

#[derive(Default, Copy, Clone, PartialEq, Eq)]
struct Symbol {
    // the byte sequence that this symbol stands for
    val: u64,

    // icl = u64 ignoredBits:16,code:12,length:4,unused:32 -- but we avoid exposing this bit-field notation
    // use a single u64 to be sure "code" is accessed with one load and can be compared with one comparison
    icl: u64,
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.val.to_ne_bytes();
        for i in 0..self.symbol_len() {
            write!(f, "{}", bytes[i as usize] as char)?;
        }
        write!(f, "\t")?;
        write!(
            f,
            "ignoredBits: {}, code: {}, length: {}",
            self.ignored_bits(),
            self.code(),
            self.symbol_len()
        )?;
        Ok(())
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.val.to_ne_bytes();
        for i in 0..self.symbol_len() {
            write!(f, "{}", bytes[i as usize] as char)?;
        }
        write!(f, "\t")?;
        write!(
            f,
            "ignoredBits: {}, code: {}, length: {}",
            self.ignored_bits(),
            self.code(),
            self.symbol_len()
        )?;
        Ok(())
    }
}

impl Symbol {
    #[inline]
    fn new() -> Self {
        Self {
            val: 0,
            icl: FSST_ICL_FREE,
        }
    }

    #[inline]
    fn from_char(c: u8, code: u16) -> Self {
        Self {
            val: c as u64,
            // in a symbol which represents a single character, 56 bits(7 bytes) are ignored, code length is 1
            icl: (1 << CODE_LEN_SHIFT_IN_ICL) | ((code as u64) << CODE_SHIFT_IN_ICL) | 56,
        }
    }

    #[inline]
    fn set_code_len(&mut self, code: u16, len: u32) {
        self.icl = ((len as u64) << CODE_LEN_SHIFT_IN_ICL)
            | ((code as u64) << CODE_SHIFT_IN_ICL)
            | ((8u64.saturating_sub(len as u64)) * 8);
    }

    #[inline(always)]
    fn symbol_len(&self) -> u32 {
        (self.icl >> CODE_LEN_SHIFT_IN_ICL) as u32
    }

    #[inline(always)]
    fn code(&self) -> u16 {
        ((self.icl >> CODE_SHIFT_IN_ICL) & FSST_CODE_MASK as u64) as u16
    }

    // ignoredBits is (8-length)*8, which is the amount of high bits to zero in the input word before comparing with the hashtable key
    // it could of course be computed from len during lookup, but storing it precomputed in some loose bits is faster
    #[inline(always)]
    fn ignored_bits(&self) -> u32 {
        (self.icl & u16::MAX as u64) as u32
    }

    #[inline(always)]
    fn first(&self) -> u8 {
        assert!(self.symbol_len() >= 1);
        (0xFF & self.val) as u8
    }

    #[inline(always)]
    fn first2(&self) -> u16 {
        assert!(self.symbol_len() >= 2);
        (0xFFFF & self.val) as u16
    }

    #[inline(always)]
    fn hash(&self) -> u64 {
        let v = 0xFFFFFF & self.val;
        fsst_hash(v)
    }

    // right is the substring follows left
    // for example, in "hello",
    // "llo" is the substring that follows "he"
    #[inline]
    fn concat(left: Self, right: Self) -> Self {
        let mut s = Self::new();
        let mut length = left.symbol_len() + right.symbol_len();
        if length > MAX_SYMBOL_LENGTH as u32 {
            length = MAX_SYMBOL_LENGTH as u32;
        }
        s.set_code_len(FSST_CODE_MASK, length);
        s.val = (right.val << (8 * left.symbol_len())) | left.val;
        s
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct SymbolKey(Symbol);

impl Hash for SymbolKey {
    // This hash algorithm follows the C++ implementation of FSST in the paper.
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut k = self.0.val;
        const M: u64 = 0xc6a4a7935bd1e995;
        const R: u32 = 47;
        let mut h: u64 = 0x8445d61a4e774912 ^ (8u64.wrapping_mul(M));
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h ^= k;
        h = h.wrapping_mul(M);
        h ^= h >> R;
        h = h.wrapping_mul(M);
        h ^= h >> R;
        state.write_u64(h);
    }
}

#[derive(Default)]
struct IdentityU64Hasher(u64);

impl Hasher for IdentityU64Hasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        let mut acc = self.0;
        for &b in bytes {
            acc = acc.wrapping_mul(0x100000001b3) ^ (b as u64);
        }
        self.0 = acc;
    }

    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

type FsstBuildHasher = BuildHasherDefault<IdentityU64Hasher>;

// Symbol that can be put in a queue, ordered on gain
#[derive(Copy, Clone, PartialEq, Eq)]
struct QSymbol {
    symbol: Symbol,
    gain: u32,
}

impl Ord for QSymbol {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain
            .cmp(&other.gain)
            .then_with(|| other.symbol.val.cmp(&self.symbol.val))
            .then_with(|| other.symbol.icl.cmp(&self.symbol.icl))
    }
}

impl PartialOrd for QSymbol {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
struct SymbolTable {
    short_codes: [u16; 65536],
    byte_codes: [u16; 256],
    symbols: [Symbol; FSST_CODE_MAX as usize],
    hash_tab: [Symbol; FSST_HASH_TAB_SIZE],
    n_symbols: u16,
    terminator: u16,
    // in a finalized symbol table, symbols are arranged by their symbol length,
    // in the order of 2, 3, 4, 5, 6, 7, 8, 1, codes < suffix_lim are 2 bytes codes that don't have a longer suffix
    suffix_lim: u16,
    len_histo: [u8; FSST_CODE_BITS as usize],
}

impl std::fmt::Display for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "A FSST SymbolTable after finalize():")?;
        writeln!(f, "n_symbols: {}", self.n_symbols)?;
        for i in 0_usize..self.n_symbols as usize {
            writeln!(f, "symbols[{}]: {}", i, self.symbols[i])?;
        }
        writeln!(f, "suffix_lim: {}", self.suffix_lim)?;
        for i in 0..FSST_CODE_BITS {
            writeln!(f, "len_histo[{}]: {}", i, self.len_histo[i as usize])?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for SymbolTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "A FSST SymbolTable before finalize():")?;
        writeln!(f, "n_symbols: {}", self.n_symbols)?;
        for i in FSST_CODE_BASE as usize..FSST_CODE_BASE as usize + self.n_symbols as usize {
            writeln!(f, "symbols[{}]: {}", i, self.symbols[i])?;
        }
        writeln!(f, "suffix_lim: {}", self.suffix_lim)?;
        for i in 0..FSST_CODE_BITS {
            writeln!(f, "len_histo[{}]: {}\n", i, self.len_histo[i as usize])?;
        }
        Ok(())
    }
}

impl SymbolTable {
    fn new() -> Self {
        let mut symbols = [Symbol::new(); FSST_CODE_MAX as usize];
        for (i, symbol) in symbols.iter_mut().enumerate().take(256) {
            *symbol = Symbol::from_char(i as u8, i as u16);
        }
        let unused = Symbol::from_char(0, FSST_CODE_MASK);
        for symbol in symbols.iter_mut().skip(256) {
            *symbol = unused;
        }
        let s = Symbol::new();
        let hash_tab = [s; FSST_HASH_TAB_SIZE];
        let mut byte_codes = [0; 256];
        for (i, byte_code) in byte_codes.iter_mut().enumerate() {
            *byte_code = i as u16;
        }
        let mut short_codes = [FSST_CODE_MASK; 65536];
        for i in 0..=65535_u16 {
            short_codes[i as usize] = i & 0xFF;
        }
        Self {
            short_codes,
            byte_codes,
            symbols,
            hash_tab,
            n_symbols: 0,
            terminator: 256,
            suffix_lim: FSST_CODE_MAX,
            len_histo: [0; FSST_CODE_BITS as usize],
        }
    }

    /// Reset a pre-finalized symbol table by reverting only the `<= 255` symbols inserted via `add`.
    fn clear(&mut self) {
        let unused = Symbol::from_char(0, FSST_CODE_MASK);
        for i in 0..self.n_symbols as usize {
            let slot = FSST_CODE_BASE as usize + i;
            let s = self.symbols[slot];
            self.symbols[slot] = unused;
            match s.symbol_len() {
                1 => {
                    let b = s.first();
                    self.byte_codes[b as usize] = b as u16;
                }
                2 => {
                    let f2 = s.first2();
                    self.short_codes[f2 as usize] = f2 & 0xFF;
                }
                _ => {
                    let idx = (s.hash() & (FSST_HASH_TAB_SIZE as u64 - 1)) as usize;
                    self.hash_tab[idx] = Symbol::new();
                }
            }
        }
        self.len_histo.fill(0);
        self.n_symbols = 0;
    }

    #[inline]
    fn hash_insert(&mut self, s: Symbol) -> bool {
        let idx = (s.hash() & (FSST_HASH_TAB_SIZE as u64 - 1)) as usize;
        let taken = self.hash_tab[idx].icl < FSST_ICL_FREE;
        if taken {
            return false; // collision in hash table
        }
        self.hash_tab[idx].icl = s.icl;
        self.hash_tab[idx].val = s.val & (u64::MAX >> s.ignored_bits());
        true
    }

    #[inline]
    fn add(&mut self, mut s: Symbol) -> bool {
        assert!(FSST_CODE_BASE + self.n_symbols < FSST_CODE_MAX);
        let len = s.symbol_len();
        s.set_code_len(FSST_CODE_BASE + self.n_symbols, len);
        if len == 1 {
            self.byte_codes[s.first() as usize] = FSST_CODE_BASE + self.n_symbols;
        } else if len == 2 {
            self.short_codes[s.first2() as usize] = FSST_CODE_BASE + self.n_symbols;
        } else if !self.hash_insert(s) {
            return false;
        }
        self.symbols[(FSST_CODE_BASE + self.n_symbols) as usize] = s;
        self.n_symbols += 1;
        self.len_histo[(len - 1) as usize] += 1;
        true
    }

    #[inline]
    fn find_longest_symbol_from_char_slice(&self, input: &[u8]) -> u16 {
        let len = input.len().min(MAX_SYMBOL_LENGTH);
        if len < 2 {
            return self.byte_codes[input[0] as usize] & FSST_CODE_MASK;
        }
        if len == 2 {
            let short_code = ((input[1] as usize) << 8) | input[0] as usize;
            if self.short_codes[short_code] >= FSST_CODE_BASE {
                return self.short_codes[short_code] & FSST_CODE_MASK;
            } else {
                return self.byte_codes[input[0] as usize] & FSST_CODE_MASK;
            }
        }
        let mut input_in_1_word = [0; 8];
        input_in_1_word[..len].copy_from_slice(&input[..len]);
        let input_in_u64 = fsst_unaligned_load_unchecked(input_in_1_word.as_ptr());
        let hash_idx = fsst_hash(input_in_u64) as usize & (FSST_HASH_TAB_SIZE - 1);
        let s_in_hash_tab = self.hash_tab[hash_idx];
        if s_in_hash_tab.icl < FSST_ICL_FREE
            && s_in_hash_tab.val == (input_in_u64 & (u64::MAX >> s_in_hash_tab.ignored_bits()))
        {
            return s_in_hash_tab.code();
        }
        self.byte_codes[input[0] as usize] & FSST_CODE_MASK
    }

    fn finalize(&mut self) {
        assert!(self.n_symbols < FSST_CODE_BASE);
        let n_syms = self.n_symbols as usize;
        let mut new_code: [u16; 256] = [0; 256];
        let mut orig_symbols: [Symbol; 256] = [Symbol::new(); 256];
        orig_symbols[..n_syms].copy_from_slice(
            &self.symbols[FSST_CODE_BASE as usize..FSST_CODE_BASE as usize + n_syms],
        );

        let mut rsum: [u8; 8] = [0; 8];
        let byte_lim = self.n_symbols - self.len_histo[0] as u16;

        rsum[0] = byte_lim as u8; // 1-byte codes are highest
        for i in 1..7 {
            rsum[i + 1] = rsum[i] + self.len_histo[i];
        }

        let mut suffix_lim = 0;
        let mut j = rsum[2];
        for i in 0..n_syms {
            let mut s1 = orig_symbols[i];
            let len = s1.symbol_len();
            if len == 2 {
                let mut has_suffix = false;
                let first2 = s1.first2();
                for (k, s2) in orig_symbols[..n_syms].iter().enumerate() {
                    if k != i && s2.symbol_len() > 2 && first2 == s2.first2() {
                        has_suffix = true;
                        break;
                    }
                }
                new_code[i] = if has_suffix {
                    suffix_lim += 1;
                    suffix_lim - 1
                } else {
                    j -= 1;
                    j as u16
                };
            } else {
                new_code[i] = rsum[(len - 1) as usize] as u16;
                rsum[(len - 1) as usize] += 1;
            }
            s1.set_code_len(new_code[i], len);
            self.symbols[new_code[i] as usize] = s1;
        }

        for i in 0..256 {
            if (self.byte_codes[i] & FSST_CODE_MASK) >= FSST_CODE_BASE {
                self.byte_codes[i] =
                    new_code[(self.byte_codes[i] & 0xFF) as usize] | (1 << CODE_LEN_SHIFT_IN_CODE);
            } else {
                self.byte_codes[i] = 511 | (1 << CODE_LEN_SHIFT_IN_CODE);
            }
        }

        // Broadcast single-byte fallback codes across all 256 blocks of `short_codes`
        for block in self.short_codes.as_chunks_mut::<256>().0 {
            *block = self.byte_codes;
        }

        // Patch only the <= 255 declared symbols into `short_codes` and `hash_tab`
        for (i, orig) in orig_symbols[..n_syms].iter().enumerate() {
            let nc = new_code[i];
            match orig.symbol_len() {
                1 => {}
                2 => {
                    self.short_codes[orig.first2() as usize] = nc | (2 << CODE_LEN_SHIFT_IN_CODE);
                }
                _ => {
                    let idx = (orig.hash() & (FSST_HASH_TAB_SIZE as u64 - 1)) as usize;
                    self.hash_tab[idx] = self.symbols[nc as usize];
                }
            }
        }
        self.suffix_lim = suffix_lim;
    }
}

const COUNTER_DIM: usize = FSST_CODE_MAX as usize;

struct Counters {
    count1: [u16; COUNTER_DIM],
    count2: Vec<u16>,
}

impl Counters {
    fn new() -> Self {
        Self {
            count1: [0; COUNTER_DIM],
            count2: vec![0; COUNTER_DIM * COUNTER_DIM],
        }
    }

    /// Reset only rows that were touched in the previous round (`count1[pos1] > 0` holds for
    /// every `pos1` whose `count2` row was incremented).
    fn reset(&mut self) {
        for (pos1, cnt1) in self.count1.iter_mut().enumerate() {
            if *cnt1 != 0 {
                *cnt1 = 0;
                let row_start = pos1 * COUNTER_DIM;
                self.count2[row_start..row_start + COUNTER_DIM].fill(0);
            }
        }
    }

    #[inline(always)]
    fn count1_set(&mut self, pos1: usize, val: u16) {
        self.count1[pos1] = val;
    }

    #[inline(always)]
    fn count1_inc(&mut self, pos1: u16) {
        let slot = &mut self.count1[pos1 as usize];
        *slot = slot.saturating_add(1);
    }

    #[inline(always)]
    fn count2_inc(&mut self, pos1: usize, pos2: usize) {
        let slot = &mut self.count2[pos1 * COUNTER_DIM + pos2];
        *slot = slot.saturating_add(1);
    }

    #[inline(always)]
    fn count1_get(&self, pos1: usize) -> u16 {
        self.count1[pos1]
    }

    #[inline(always)]
    fn count2_row(&self, pos1: usize, limit: usize) -> &[u16] {
        let start = pos1 * COUNTER_DIM;
        &self.count2[start..start + limit]
    }
}

#[inline(always)]
fn is_escape_code(pos: u16) -> bool {
    pos < FSST_CODE_BASE
}

// Select roughly FSST_SAMPLETARGET bytes, preserving complete strings.
fn make_sample<T: OffsetSizeTrait>(
    in_buf: &[u8],
    offsets: &[T],
    rng: &mut impl Rng,
) -> (Vec<u8>, Vec<T>) {
    let total_size = in_buf.len();
    if total_size <= FSST_SAMPLETARGET {
        return (in_buf.to_vec(), offsets.to_vec());
    }
    let mut sample_buf = Vec::with_capacity(FSST_SAMPLEMAXSZ);
    let mut sample_offsets = Vec::with_capacity(FSST_SAMPLETARGET + 2);

    sample_offsets.push(T::zero());
    // Each nonempty draw contributes at least one byte. Limit unsuccessful
    // draws too: sparse inputs must not allocate offsets or retry indefinitely.
    // A single source sweep completes the sample if random draws miss its data.
    let random_indices =
        (0..FSST_SAMPLETARGET).map(|_| rng.random_range(0..offsets.len()) % (offsets.len() - 1));
    for index in random_indices.chain(0..offsets.len() - 1) {
        let start = offsets[index].as_usize();
        let end = offsets[index + 1].as_usize();
        if start == end {
            continue;
        }
        sample_buf.extend_from_slice(&in_buf[start..end]);
        sample_offsets.push(T::from_usize(sample_buf.len()).unwrap());
        if sample_buf.len() >= FSST_SAMPLETARGET {
            break;
        }
    }
    sample_offsets.push(T::from_usize(sample_buf.len()).unwrap());
    (sample_buf, sample_offsets)
}

#[inline(always)]
fn match_next_symbol_in_sample(st: &SymbolTable, word: &[u8], curr: usize) -> (u16, u32) {
    if word.len() > 7 && curr < word.len() - 7 {
        let mut this_64_bit_word: u64 = fsst_unaligned_load_unchecked(word[curr..].as_ptr());
        let code = this_64_bit_word & 0xFFFFFF;
        let idx = fsst_hash(code) as usize & (FSST_HASH_TAB_SIZE - 1);
        let s: Symbol = st.hash_tab[idx];
        let short_code = st.short_codes[(this_64_bit_word & 0xFFFF) as usize] & FSST_CODE_MASK;
        this_64_bit_word &= 0xFFFFFFFFFFFFFFFF >> s.icl as u8;
        if (s.icl < FSST_ICL_FREE) & (s.val == this_64_bit_word) {
            (s.code(), s.symbol_len())
        } else if short_code >= FSST_CODE_BASE {
            (short_code, 2)
        } else {
            (
                st.byte_codes[(this_64_bit_word & 0xFF) as usize] & FSST_CODE_MASK,
                1,
            )
        }
    } else {
        let curr_code = st.find_longest_symbol_from_char_slice(&word[curr..]);
        (curr_code, st.symbols[curr_code as usize].symbol_len())
    }
}

// build_symbol_table constructs a symbol table from a sample of the input
fn build_symbol_table<T: OffsetSizeTrait>(
    sample_buf: &[u8],
    sample_offsets: &[T],
) -> io::Result<Box<SymbolTable>> {
    let mut st = Box::new(SymbolTable::new());
    let mut best_symbols = [Symbol::new(); FSST_CODE_BASE as usize];
    let mut best_n_symbols: u16 = 0;
    // worst case (everything exception), will be updated later
    let mut best_gain = T::zero() - T::from_usize(FSST_SAMPLEMAXSZ).unwrap();

    let mut byte_histo = [0usize; 256];
    for &c in sample_buf {
        byte_histo[c as usize] += 1;
    }
    let mut curr_min_histo = FSST_SAMPLEMAXSZ;
    for (i, &this_byte_histo) in byte_histo.iter().enumerate() {
        if this_byte_histo < curr_min_histo {
            curr_min_histo = this_byte_histo;
            st.terminator = i as u16;
        }
    }

    let mut counters = Counters::new();
    let mut candidates: HashMap<SymbolKey, u32, FsstBuildHasher> =
        HashMap::with_capacity_and_hasher(2048, FsstBuildHasher::default());
    let mut pq_vec: Vec<QSymbol> = Vec::with_capacity(2048);

    // Compress sample, and compute (pair-)frequencies for rounds that build a next table
    let compress_count = |st: &SymbolTable, counters: &mut Counters| -> usize {
        let mut gain = 0usize;
        for i in 1..sample_offsets.len() {
            let start = sample_offsets[i - 1].as_usize();
            let end = sample_offsets[i].as_usize();
            if start == end {
                continue;
            }
            let word = &sample_buf[start..end];

            let mut prev_code = st.find_longest_symbol_from_char_slice(word);
            let mut prev_len = st.symbols[prev_code as usize].symbol_len() as usize;
            let mut curr = prev_len;
            gain += prev_len.saturating_sub(1 + usize::from(is_escape_code(prev_code)));

            while curr < word.len() {
                counters.count1_inc(prev_code);
                if prev_len != 1 {
                    counters.count1_inc(word[curr] as u16);
                }

                let (curr_code, symbol_len) = match_next_symbol_in_sample(st, word, curr);
                let symbol_len_usize = symbol_len as usize;
                gain += symbol_len_usize.saturating_sub(1 + usize::from(is_escape_code(curr_code)));

                counters.count2_inc(prev_code as usize, curr_code as usize);
                if symbol_len > 1 {
                    counters.count2_inc(prev_code as usize, word[curr] as usize);
                }

                curr += symbol_len_usize;
                prev_code = curr_code;
                prev_len = symbol_len_usize;
            }
            counters.count1_inc(prev_code);
        }
        gain
    };

    // Final round (frac == 128) only evaluates `gain` for the last candidate table; no further
    // `make_table` pass runs after it, so counter updates are unnecessary.
    let compress_gain_only = |st: &SymbolTable| -> usize {
        let mut gain = 0usize;
        for i in 1..sample_offsets.len() {
            let start = sample_offsets[i - 1].as_usize();
            let end = sample_offsets[i].as_usize();
            if start == end {
                continue;
            }
            let word = &sample_buf[start..end];

            let mut prev_code = st.find_longest_symbol_from_char_slice(word);
            let mut curr = st.symbols[prev_code as usize].symbol_len() as usize;
            gain += curr.saturating_sub(1 + usize::from(is_escape_code(prev_code)));

            while curr < word.len() {
                let (curr_code, symbol_len) = match_next_symbol_in_sample(st, word, curr);
                let symbol_len_usize = symbol_len as usize;
                gain += symbol_len_usize.saturating_sub(1 + usize::from(is_escape_code(curr_code)));
                curr += symbol_len_usize;
                prev_code = curr_code;
            }
            let _ = prev_code;
        }
        gain
    };

    let make_table = |st: &mut SymbolTable,
                      counters: &mut Counters,
                      sample_frac: usize,
                      candidates: &mut HashMap<SymbolKey, u32, FsstBuildHasher>,
                      pq_vec: &mut Vec<QSymbol>| {
        candidates.clear();
        counters.count1_set(st.terminator as usize, u16::MAX);
        let min_count = (5 * sample_frac as u64) / 128;

        let add_or_inc =
            |cands: &mut HashMap<SymbolKey, u32, FsstBuildHasher>, s: Symbol, count: u64| {
                if count < min_count {
                    return;
                }
                let gain = (count * s.symbol_len() as u64) as u32;
                cands
                    .entry(SymbolKey(s))
                    .and_modify(|existing| *existing += gain)
                    .or_insert(gain);
            };

        let limit = FSST_CODE_BASE as usize + st.n_symbols as usize;
        let term_u8 = st.terminator as u8;
        for pos1 in 0..limit {
            let cnt1 = counters.count1_get(pos1);
            if cnt1 == 0 {
                continue;
            }
            // heuristic: promoting single-byte symbols (*8) helps reduce exception rates and increases [de]compression speed
            let s1 = st.symbols[pos1];
            add_or_inc(
                candidates,
                s1,
                if s1.symbol_len() == 1 { 8 } else { 1 } * cnt1 as u64,
            );
            if s1.first() == term_u8 || s1.symbol_len() == MAX_SYMBOL_LENGTH as u32 {
                continue;
            }
            let row = counters.count2_row(pos1, limit);
            for (pos2, &cnt2) in row.iter().enumerate() {
                if cnt2 == 0 {
                    continue;
                }
                let s2 = st.symbols[pos2];
                // multi-byte symbols cannot contain the terminator byte
                if s2.first() != term_u8 {
                    let s3 = Symbol::concat(s1, s2);
                    add_or_inc(candidates, s3, cnt2 as u64);
                }
            }
        }

        pq_vec.clear();
        pq_vec.extend(
            candidates
                .drain()
                .map(|(k, gain)| QSymbol { symbol: k.0, gain }),
        );
        pq_vec.sort_unstable_by(|a, b| b.cmp(a));

        st.clear();
        for q in pq_vec.iter() {
            if st.n_symbols >= 255 {
                break;
            }
            st.add(q.symbol);
        }
    };

    for (round_idx, frac) in [8usize, 38, 68, 98, 108].into_iter().enumerate() {
        if round_idx > 0 {
            counters.reset();
        }
        let gain = T::from_usize(compress_count(&st, &mut counters)).unwrap();
        if gain >= best_gain {
            best_gain = gain;
            best_n_symbols = st.n_symbols;
            let n = best_n_symbols as usize;
            best_symbols[..n]
                .copy_from_slice(&st.symbols[FSST_CODE_BASE as usize..FSST_CODE_BASE as usize + n]);
        }
        make_table(&mut st, &mut counters, frac, &mut candidates, &mut pq_vec);
    }

    // Final evaluation round (frac == 128)
    let final_gain = T::from_usize(compress_gain_only(&st)).unwrap();
    if final_gain < best_gain {
        st.clear();
        for &sym in &best_symbols[..best_n_symbols as usize] {
            let added = st.add(sym);
            debug_assert!(added, "replaying collision-free best symbols must succeed");
        }
    }

    st.finalize(); // renumber codes for more efficient compression
    if st.n_symbols == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Fsst failed to build symbol table, input len: {}, input_offsets len: {}",
                sample_buf.len(),
                sample_offsets.len()
            ),
        ));
    }
    Ok(st)
}

#[inline(always)]
unsafe fn compress_step(
    st: &SymbolTable,
    in_ptr: *const u8,
    out_ptr: *mut u8,
    out_curr: &mut usize,
) -> usize {
    let word = fsst_unaligned_load_unchecked(in_ptr);
    let short_code = st.short_codes[(word & 0xFFFF) as usize];
    let word_first_3_byte = word & 0xFFFFFF;
    let idx = fsst_hash(word_first_3_byte) as usize & (FSST_HASH_TAB_SIZE - 1);
    let s = st.hash_tab[idx];
    // Speculatively write escaped literal byte; overwritten on the next step unless `code` has bit 8 set.
    *out_ptr.add(*out_curr + 1) = word as u8;
    let code = if s.icl < FSST_ICL_FREE && s.val == (word & (u64::MAX >> (s.icl & 0xFFFF))) {
        (s.icl >> 16) as u16
    } else {
        short_code
    };
    *out_ptr.add(*out_curr) = code as u8;
    *out_curr += 1 + ((code & 256) >> 8) as usize;
    (code >> 12) as usize
}

fn compress_bulk<T: OffsetSizeTrait>(
    st: &SymbolTable,
    strs: &[u8],
    offsets: &[T],
    out: &mut Vec<u8>,
    out_offsets: &mut Vec<T>,
) -> io::Result<()> {
    if offsets.is_empty() {
        out.clear();
        out_offsets.clear();
        return Ok(());
    }

    // Worst-case expansion is 2 bytes per input byte (every byte escaped), plus 1 byte of
    // speculative lookahead on the final symbol write.
    let needed_out = strs
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_add(2))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "FSST input size overflow"))?;
    if out.capacity() < needed_out {
        out.reserve(needed_out - out.len());
    }
    if out_offsets.capacity() < offsets.len() {
        out_offsets.reserve(offsets.len() - out_offsets.len());
    }

    let strs_ptr = strs.as_ptr();
    let out_ptr = out.as_mut_ptr();
    let out_offsets_ptr = out_offsets.as_mut_ptr();
    let terminator = st.terminator as u8;
    let mut tail = [0u8; 16];
    let mut out_curr = 0usize;

    // SAFETY: `out_offsets` has capacity >= `offsets.len()`.
    unsafe {
        out_offsets_ptr.write(T::zero());
    }

    for (i, window) in offsets.windows(2).enumerate() {
        let mut in_curr = window[0].as_usize();
        let end_curr = window[1].as_usize();

        // Direct zero-copy fast path while at least 8 bytes remain in the current string.
        // Because every symbol has length in 1..=8, `in_curr` never advances past `end_curr`.
        while in_curr + 8 <= end_curr {
            // SAFETY: `in_curr + 8 <= end_curr <= strs.len()` and `out_curr + 2 <= needed_out <= out.capacity()`.
            unsafe {
                in_curr += compress_step(st, strs_ptr.add(in_curr), out_ptr, &mut out_curr);
            }
        }

        // Remaining 1..=7 bytes: copy once into the small stack tail buffer and append `terminator`.
        // Multi-byte symbols in `SymbolTable` never contain `terminator`, so matches never cross `rem`.
        if in_curr < end_curr {
            let rem = end_curr - in_curr;
            tail[..rem].copy_from_slice(&strs[in_curr..end_curr]);
            tail[rem] = terminator;
            let mut t_curr = 0usize;
            let tail_ptr = tail.as_ptr();
            while t_curr < rem {
                // SAFETY: `t_curr < rem <= 7`, so `tail_ptr.add(t_curr)` has at least 9 readable bytes in `tail`.
                unsafe {
                    t_curr += compress_step(st, tail_ptr.add(t_curr), out_ptr, &mut out_curr);
                }
            }
        }

        let encoded_off = encode_offset(out_curr)?;
        // SAFETY: `i + 1 < offsets.len() <= out_offsets.capacity()`.
        unsafe {
            out_offsets_ptr.add(i + 1).write(encoded_off);
        }
    }

    // SAFETY: `0..out_curr` and `0..offsets.len()` were written monotonically above.
    unsafe {
        out.set_len(out_curr);
        out_offsets.set_len(offsets.len());
    }
    Ok(())
}

#[inline]
fn offset_to_usize<T: OffsetSizeTrait>(offset: T) -> io::Result<usize> {
    offset.to_usize().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "FSST offset (as usize {}) is negative or exceeds {}",
                offset.as_usize(),
                T::MAX_OFFSET
            ),
        )
    })
}

fn validate_offsets<T: OffsetSizeTrait>(offsets: &[T], compressed_len: usize) -> io::Result<()> {
    let Some((first, rest)) = offsets.split_first() else {
        return Ok(());
    };
    let mut previous = offset_to_usize(*first)?;
    if previous > compressed_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "FSST offset[0] = {previous} is out of bounds for compressed buffer of length {compressed_len}"
            ),
        ));
    }
    for (index, offset) in rest.iter().enumerate() {
        let current = offset_to_usize(*offset)?;
        let position = index + 1;
        if current < previous {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("FSST offset at position {position} decreases: {current} < {previous}"),
            ));
        }
        if current > compressed_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "FSST offset at position {position} = {current} is out of bounds for compressed buffer of length {compressed_len}"
                ),
            ));
        }
        previous = current;
    }
    Ok(())
}

#[inline]
fn encode_offset<T: OffsetSizeTrait>(value: usize) -> io::Result<T> {
    T::from_usize(value).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("FSST decompressed size {value} does not fit in the offset type"),
        )
    })
}

/// The output buffer's allocation, written through a raw pointer so the decoder can fill
/// spare capacity that was never initialized.
#[derive(Clone, Copy)]
struct OutBuf {
    ptr: *mut u8,
    capacity: usize,
}

#[inline(always)]
fn write_symbol(out: OutBuf, out_curr: usize, symbol: u64) {
    debug_assert!(
        out_curr
            .checked_add(8)
            .is_some_and(|end| end <= out.capacity),
        "FSST symbol write at {out_curr} overflows buffer of capacity {}",
        out.capacity
    );
    // SAFETY: `FsstDecompressor::try_new` rejected any declared `lens[i]` outside 1..=8 and left
    // undeclared slots at 0, so every code advances `out_curr` by at most 8. Combined with
    // the 8x output-capacity check, `out_curr + 8 <= out.capacity` holds for every write.
    unsafe {
        ptr::write_unaligned(out.ptr.add(out_curr) as *mut u64, symbol);
    }
}

#[inline(always)]
fn store_out_byte(out: OutBuf, out_curr: usize, byte: u8) {
    debug_assert!(
        out_curr < out.capacity,
        "FSST literal write at {out_curr} overflows buffer of capacity {}",
        out.capacity
    );
    // SAFETY: same 8x + `lens[code] <= 8` proof as `write_symbol`.
    unsafe {
        out.ptr.add(out_curr).write(byte);
    }
}

/// Consume `FSST_ESC` at `in_curr` and emit its payload byte.
///
/// Returns `false` when the payload would leave the current value interval
/// `[in_curr, in_end)`, including a dangling escape at the last byte.
#[inline(always)]
fn emit_escape(
    compressed_strs: &[u8],
    in_curr: &mut usize,
    in_end: usize,
    out: OutBuf,
    out_curr: &mut usize,
) -> bool {
    let payload_index = *in_curr + 1;
    if payload_index >= in_end {
        return false;
    }
    store_out_byte(out, *out_curr, compressed_strs[payload_index]);
    *out_curr += 1;
    *in_curr += 2;
    true
}

fn missing_escape_payload_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "FSST escape is missing a payload byte inside the current value",
    )
}

fn decompress_bulk<T: OffsetSizeTrait>(
    decoder: &FsstDecompressor,
    compressed_strs: &[u8],
    offsets: &[T],
    out: &mut Vec<u8>,
    out_offsets: &mut Vec<T>,
) -> io::Result<()> {
    if offsets.is_empty() {
        out.clear();
        out_offsets.clear();
        return Ok(());
    }

    let compressed_len = compressed_strs.len();
    let mut in_curr_prev = offset_to_usize(offsets[0])?;
    if in_curr_prev > compressed_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "FSST offset[0] = {in_curr_prev} is out of bounds for compressed buffer of length {compressed_len}"
            ),
        ));
    }

    let symbols = &decoder.symbols;
    let lens = &decoder.lens;
    let in_ptr = compressed_strs.as_ptr();
    let out_buf = OutBuf {
        ptr: out.as_mut_ptr(),
        capacity: out.capacity(),
    };
    let out_offsets_ptr = out_offsets.as_mut_ptr();
    let corrupt_escape = Cell::new(false);

    let decompress_one = |mut in_curr: usize, in_end: usize, out_curr: &mut usize| {
        // 8-byte SWAR fast path: detect `0xFF` (`FSST_ESC`) across 8 codes simultaneously.
        while in_curr + 8 <= in_end {
            // SAFETY: `in_curr + 8 <= in_end <= compressed_len` was validated before calling `decompress_one`.
            let block8 =
                u64::from_le(unsafe { ptr::read_unaligned(in_ptr.add(in_curr) as *const u64) });
            let escape_mask = (!block8).wrapping_sub(0x0101_0101_0101_0101u64)
                & block8
                & 0x8080_8080_8080_8080u64;
            if escape_mask == 0 {
                let c0 = (block8 as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c0]);
                *out_curr += lens[c0] as usize;

                let c1 = ((block8 >> 8) as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c1]);
                *out_curr += lens[c1] as usize;

                let c2 = ((block8 >> 16) as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c2]);
                *out_curr += lens[c2] as usize;

                let c3 = ((block8 >> 24) as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c3]);
                *out_curr += lens[c3] as usize;

                let c4 = ((block8 >> 32) as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c4]);
                *out_curr += lens[c4] as usize;

                let c5 = ((block8 >> 40) as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c5]);
                *out_curr += lens[c5] as usize;

                let c6 = ((block8 >> 48) as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c6]);
                *out_curr += lens[c6] as usize;

                let c7 = ((block8 >> 56) as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c7]);
                *out_curr += lens[c7] as usize;

                in_curr += 8;
            } else {
                let first_esc = (escape_mask.trailing_zeros() >> 3) as usize;
                let mut shifted = block8;
                for _ in 0..first_esc {
                    let code = (shifted as u8) as usize;
                    write_symbol(out_buf, *out_curr, symbols[code]);
                    *out_curr += lens[code] as usize;
                    shifted >>= 8;
                }
                in_curr += first_esc;
                if first_esc < 7 {
                    store_out_byte(out_buf, *out_curr, (shifted >> 8) as u8);
                    *out_curr += 1;
                    in_curr += 2;
                } else if !emit_escape(compressed_strs, &mut in_curr, in_end, out_buf, out_curr) {
                    corrupt_escape.set(true);
                    return;
                }
            }
        }

        // 4-byte SWAR cleanup for remaining 4..=7 bytes
        if in_curr + 4 <= in_end {
            // SAFETY: `in_curr + 4 <= in_end <= compressed_len`.
            let block4 =
                u32::from_le(unsafe { ptr::read_unaligned(in_ptr.add(in_curr) as *const u32) });
            let escape_mask = (!block4).wrapping_sub(0x0101_0101u32) & block4 & 0x8080_8080u32;
            if escape_mask == 0 {
                let c0 = (block4 as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c0]);
                *out_curr += lens[c0] as usize;

                let c1 = ((block4 >> 8) as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c1]);
                *out_curr += lens[c1] as usize;

                let c2 = ((block4 >> 16) as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c2]);
                *out_curr += lens[c2] as usize;

                let c3 = ((block4 >> 24) as u8) as usize;
                write_symbol(out_buf, *out_curr, symbols[c3]);
                *out_curr += lens[c3] as usize;

                in_curr += 4;
            } else {
                let first_esc = (escape_mask.trailing_zeros() >> 3) as usize;
                let mut shifted = block4;
                for _ in 0..first_esc {
                    let code = (shifted as u8) as usize;
                    write_symbol(out_buf, *out_curr, symbols[code]);
                    *out_curr += lens[code] as usize;
                    shifted >>= 8;
                }
                in_curr += first_esc;
                if first_esc < 3 {
                    store_out_byte(out_buf, *out_curr, (shifted >> 8) as u8);
                    *out_curr += 1;
                    in_curr += 2;
                } else if !emit_escape(compressed_strs, &mut in_curr, in_end, out_buf, out_curr) {
                    corrupt_escape.set(true);
                    return;
                }
            }
        }

        // Scalar tail for remaining 0..=3 bytes
        while in_curr < in_end {
            let b = compressed_strs[in_curr];
            if b == FSST_ESC {
                if !emit_escape(compressed_strs, &mut in_curr, in_end, out_buf, out_curr) {
                    corrupt_escape.set(true);
                    return;
                }
            } else {
                let code = b as usize;
                write_symbol(out_buf, *out_curr, symbols[code]);
                in_curr += 1;
                *out_curr += lens[code] as usize;
            }
        }
    };

    let mut out_curr = 0usize;
    // SAFETY: `out_offsets` capacity >= `offsets.len()` is checked by callers before `decompress_bulk`.
    unsafe {
        out_offsets_ptr.write(encode_offset(0)?);
    }

    for (index, &raw_end) in offsets[1..].iter().enumerate() {
        let position = index + 1;
        let in_end = offset_to_usize(raw_end)?;
        if in_end < in_curr_prev {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("FSST offset at position {position} decreases: {in_end} < {in_curr_prev}"),
            ));
        }
        if in_end > compressed_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "FSST offset at position {position} = {in_end} is out of bounds for compressed buffer of length {compressed_len}"
                ),
            ));
        }
        decompress_one(in_curr_prev, in_end, &mut out_curr);
        if corrupt_escape.get() {
            return Err(missing_escape_payload_error());
        }
        let encoded_off = encode_offset(out_curr)?;
        // SAFETY: `position < offsets.len() <= out_offsets.capacity()`.
        unsafe {
            out_offsets_ptr.add(position).write(encoded_off);
        }
        in_curr_prev = in_end;
    }

    // SAFETY: every step wrote at least the bytes it advanced `out_curr` by, so `[0, out_curr)`
    // is initialized and `out_curr <= out.capacity()`; all `offsets.len()` entries of `out_offsets`
    // were written above.
    unsafe {
        out.set_len(out_curr);
        out_offsets.set_len(offsets.len());
    }
    Ok(())
}

fn export_symbol_table(
    st: Option<&SymbolTable>,
    encoder_switch: bool,
    symbol_table_buf: &mut [u8],
) {
    let (suffix_lim, terminator, n_symbols) = match st {
        Some(st) => (st.suffix_lim, st.terminator, st.n_symbols),
        None => (FSST_CODE_MAX, 256, 0),
    };
    let st_info: u64 = FSST_MAGIC
        | ((encoder_switch as u64) << 24)
        | (((suffix_lim & 255) as u64) << 16)
        | (((terminator & 255) as u64) << 8)
        | ((n_symbols & 255) as u64);

    let st_info_bytes = st_info.to_ne_bytes();
    let mut pos = 0;
    symbol_table_buf[pos..pos + 8].copy_from_slice(&st_info_bytes);
    pos += 8;

    if let Some(st) = st {
        let n = st.n_symbols as usize;
        for i in 0..n {
            let s_bytes = st.symbols[i].val.to_ne_bytes();
            symbol_table_buf[pos..pos + 8].copy_from_slice(&s_bytes);
            pos += 8;
        }
        for i in 0..n {
            symbol_table_buf[pos + i] = st.symbols[i].symbol_len() as u8;
        }
    }
}

const FSST_CORRUPT: u64 = 32774747032022883; // 7-byte number in little endian containing "corrupt"

/// Pre-parsed FSST symbol table that can be reused across many compressed chunks on a page
/// without re-validating and unpacking the 2,312-byte symbol table on every chunk.
#[derive(Debug, Clone)]
pub struct FsstDecompressor {
    lens: [u8; 256],
    symbols: [u64; 256],
    decoder_switch_on: bool,
}

impl FsstDecompressor {
    pub fn try_new(symbol_table: &[u8]) -> io::Result<Self> {
        if symbol_table.len() != FSST_SYMBOL_TABLE_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "the symbol table buffer for FSST decoder must have size {}",
                    FSST_SYMBOL_TABLE_SIZE
                ),
            ));
        }

        let st_info = u64::from_ne_bytes(symbol_table[..8].try_into().unwrap());
        if st_info & FSST_MAGIC != FSST_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "the input buffer is not a valid FSST compressed data",
            ));
        }

        let decoder_switch_on = (st_info & (1 << 24)) != 0;
        let symbol_num = (st_info & 255) as usize;
        let mut symbols = [FSST_CORRUPT; 256];
        let mut lens = [0u8; 256];

        let mut pos = 8;
        for slot in symbols.iter_mut().take(symbol_num) {
            *slot = fsst_unaligned_load_unchecked(symbol_table[pos..].as_ptr());
            pos += 8;
        }
        for (i, slot) in lens.iter_mut().enumerate().take(symbol_num) {
            let len = symbol_table[pos];
            if !(1..=MAX_SYMBOL_LENGTH as u8).contains(&len) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "FSST symbol length at index {i} is {len}, expected 1..={MAX_SYMBOL_LENGTH}"
                    ),
                ));
            }
            *slot = len;
            pos += 1;
        }

        Ok(Self {
            lens,
            symbols,
            decoder_switch_on,
        })
    }

    /// Decompress into newly allocated buffers without zero-initializing spare capacity.
    pub fn decompress_into<T: OffsetSizeTrait>(
        &self,
        in_buf: &[u8],
        in_offsets_buf: &[T],
    ) -> io::Result<(Vec<u8>, Vec<T>)> {
        if !self.decoder_switch_on {
            validate_offsets(in_offsets_buf, in_buf.len())?;
            return Ok((in_buf.to_vec(), in_offsets_buf.to_vec()));
        }
        let needed_cap = in_buf.len().checked_mul(8).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "output buffer too small for FSST decoder",
            )
        })?;
        let mut out_buf = Vec::with_capacity(needed_cap);
        let mut out_offsets_buf = Vec::with_capacity(in_offsets_buf.len());
        decompress_bulk(
            self,
            in_buf,
            in_offsets_buf,
            &mut out_buf,
            &mut out_offsets_buf,
        )?;
        Ok((out_buf, out_offsets_buf))
    }

    pub fn decompress<T: OffsetSizeTrait>(
        &self,
        in_buf: &[u8],
        in_offsets_buf: &[T],
        out_buf: &mut Vec<u8>,
        out_offsets_buf: &mut Vec<T>,
    ) -> io::Result<()> {
        let out_capacity = out_buf.capacity();
        if self.decoder_switch_on
            && in_buf
                .len()
                .checked_mul(8)
                .is_none_or(|needed| needed > out_capacity)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "output buffer too small for FSST decoder",
            ));
        }
        if !self.decoder_switch_on && in_buf.len() > out_capacity {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "output buffer too small for FSST decoder",
            ));
        }
        if in_offsets_buf.len() > out_offsets_buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "output offsets buffer ({}) too small for FSST decoder (need at least {})",
                    out_offsets_buf.len(),
                    in_offsets_buf.len()
                ),
            ));
        }

        if !self.decoder_switch_on {
            validate_offsets(in_offsets_buf, in_buf.len())?;
            out_buf.clear();
            out_buf.extend_from_slice(in_buf);
            out_offsets_buf.resize(in_offsets_buf.len(), T::zero());
            out_offsets_buf.copy_from_slice(in_offsets_buf);
            return Ok(());
        }

        decompress_bulk(self, in_buf, in_offsets_buf, out_buf, out_offsets_buf)
    }
}

/// Compress `in_buf` / `in_offsets_buf` into freshly allocated output buffers without zero-filling
/// worst-case 2x headroom upfront.
pub fn compress_into<T: OffsetSizeTrait>(
    symbol_table: &mut [u8],
    in_buf: &[u8],
    in_offsets_buf: &[T],
) -> io::Result<(Vec<u8>, Vec<T>)> {
    if symbol_table.len() != FSST_SYMBOL_TABLE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "the symbol table buffer for FSST encoder must have size {}",
                FSST_SYMBOL_TABLE_SIZE
            ),
        ));
    }

    validate_offsets(in_offsets_buf, in_buf.len())?;

    if in_offsets_buf.len() <= 1 {
        export_symbol_table(None, false, symbol_table);
        return Ok((Vec::new(), vec![T::zero(); in_offsets_buf.len()]));
    }

    if in_buf.len() < FSST_LEAST_INPUT_SIZE {
        export_symbol_table(None, false, symbol_table);
        return Ok((in_buf.to_vec(), in_offsets_buf.to_vec()));
    }

    let mut rng = SmallRng::seed_from_u64(0x4653_5354_2026_0001 ^ (in_buf.len() as u64));
    let st = if in_buf.len() <= FSST_SAMPLETARGET {
        build_symbol_table(in_buf, in_offsets_buf)?
    } else {
        let (sample, sample_offsets) = make_sample(in_buf, in_offsets_buf, &mut rng);
        build_symbol_table(&sample, &sample_offsets)?
    };
    export_symbol_table(Some(&st), true, symbol_table);

    let mut out_buf = Vec::with_capacity(in_buf.len().saturating_mul(2).saturating_add(2));
    let mut out_offsets_buf = Vec::with_capacity(in_offsets_buf.len());
    compress_bulk(
        &st,
        in_buf,
        in_offsets_buf,
        &mut out_buf,
        &mut out_offsets_buf,
    )?;
    Ok((out_buf, out_offsets_buf))
}

/// This is the public API for the FSST compression, when the in_buf is less than FSST_LEAST_INPUT_SIZE, we put the FSST_MAGIC header and then copy the input to the output
/// we check to make sure the out_buf's size is at least the same as the in_buf's size, otherwise Err is returned, this is actually
/// risky as in some randomly generated data, the output size can be larger than the input size.
/// the out_offsets_buf should be at least the same size as the in_offsets_buf, otherwise Err is returned
/// the symbol_table is used to store the symbol table created by `compression`, it's size should be FSST_SYMBOL_TABLE_SIZE
/// after compression, the first 64 bits of the output buffer is the fsst header:
/// from most significant bit to least significant bit:
/// FSST_MAGIC| encoder_switch |    suffix_lim | terminator | n_symbols
/// | 32-bits |         8 bits |        8 bits |     8 bits | 8 bits
/// then followed by the compressed data
pub fn compress<T: OffsetSizeTrait>(
    symbol_table: &mut [u8],
    in_buf: &[u8],
    in_offsets_buf: &[T],
    out_buf: &mut Vec<u8>,
    out_offsets_buf: &mut Vec<T>,
) -> io::Result<()> {
    if symbol_table.len() != FSST_SYMBOL_TABLE_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "the symbol table buffer for FSST encoder must have size {}",
                FSST_SYMBOL_TABLE_SIZE
            ),
        ));
    }

    validate_offsets(in_offsets_buf, in_buf.len())?;

    if in_offsets_buf.len() <= 1 {
        if in_offsets_buf.len() > out_offsets_buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "output offsets buffer ({}) too small for FSST encoder (need at least {})",
                    out_offsets_buf.len(),
                    in_offsets_buf.len()
                ),
            ));
        }
        export_symbol_table(None, false, symbol_table);
        out_buf.clear();
        out_offsets_buf.clear();
        out_offsets_buf.resize(in_offsets_buf.len(), T::zero());
        return Ok(());
    }

    if in_buf.len() < FSST_LEAST_INPUT_SIZE {
        export_symbol_table(None, false, symbol_table);
        out_buf.resize(in_buf.len(), 0);
        out_buf.copy_from_slice(in_buf);
        out_offsets_buf.resize(in_offsets_buf.len(), T::zero());
        out_offsets_buf.copy_from_slice(in_offsets_buf);
        return Ok(());
    }

    if in_buf.len() > out_buf.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "output buffer ({}) too small for FSST encoder (need at least {})",
                out_buf.len(),
                in_buf.len()
            ),
        ));
    }
    if in_offsets_buf.len() > out_offsets_buf.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "output offsets buffer ({}) too small for FSST encoder (need at least {})",
                out_offsets_buf.len(),
                in_offsets_buf.len()
            ),
        ));
    }

    let mut rng = SmallRng::seed_from_u64(0x4653_5354_2026_0001 ^ (in_buf.len() as u64));
    let st = if in_buf.len() <= FSST_SAMPLETARGET {
        build_symbol_table(in_buf, in_offsets_buf)?
    } else {
        let (sample, sample_offsets) = make_sample(in_buf, in_offsets_buf, &mut rng);
        build_symbol_table(&sample, &sample_offsets)?
    };
    export_symbol_table(Some(&st), true, symbol_table);
    compress_bulk(&st, in_buf, in_offsets_buf, out_buf, out_offsets_buf)
}

pub fn decompress<T: OffsetSizeTrait>(
    symbol_table: &[u8],
    in_buf: &[u8],
    in_offsets_buf: &[T],
    out_buf: &mut Vec<u8>,
    out_offsets_buf: &mut Vec<T>,
) -> io::Result<()> {
    let decoder = FsstDecompressor::try_new(symbol_table)?;
    decoder.decompress(in_buf, in_offsets_buf, out_buf, out_offsets_buf)
}

#[cfg(test)]
mod tests {
    use crate::fsst::*;
    use arrow_array::StringArray;

    const TEST_PARAGRAPH: &str = "ACT I. Scene I.
    Elsinore. A platform before the Castle.

    Enter two Sentinels-[first,] Francisco, [who paces up and down
    at his post; then] Bernardo, [who approaches him].

        Ber. Who's there.?
        Fran. Nay, answer me. Stand and unfold yourself.
        Ber. Long live the King!
        Fran. Bernardo?
        Ber. He.
        Fran. You come most carefully upon your hour.
        Ber. 'Tis now struck twelve. Get thee to bed, Francisco.
        Fran. For this relief much thanks. 'Tis bitter cold,
        And I am sick at heart.
        Ber. Have you had quiet guard?
        Fran. Not a mouse stirring.
        Ber. Well, good night.
        If you do meet Horatio and Marcellus,
        The rivals of my watch, bid them make haste.
        Enter Horatio and Marcellus.

        Fran. I think I hear them. Stand, ho! Who is there?
        Hor. Friends to this ground.
        Mar. And liegemen to the Dane.
        Fran. Give you good night.
        Mar. O, farewell, honest soldier.
            Who hath reliev'd you?
        Fran. Bernardo hath my place.
            Give you good night.                                   Exit.
        Mar. Holla, Bernardo!
        Ber. Say-
            What, is Horatio there ?
        Hor. A piece of him.
        Ber. Welcome, Horatio. Welcome, good Marcellus.
        Mar. What, has this thing appear'd again to-night?
        Ber. I have seen nothing.
        Mar. Horatio says 'tis but our fantasy,
            And will not let belief take hold of him
            Touching this dreaded sight, twice seen of us.
            Therefore I have entreated him along,
            With us to watch the minutes of this night,
            That, if again this apparition come,
            He may approve our eyes and speak to it.
        Hor. Tush, tush, 'twill not appear.
        Ber. Sit down awhile,
            And let us once again assail your ears,
            That are so fortified against our story,
            What we two nights have seen.
        Hor. Well, sit we down,
            And let us hear Bernardo speak of this.
        Ber. Last night of all,
            When yond same star that's westward from the pole
            Had made his course t' illume that part of heaven
            Where now it burns, Marcellus and myself,
            The bell then beating one-

                                Enter Ghost.

        Mar. Peace! break thee off! Look where it comes again!
        Ber. In the same figure, like the King that's dead.
        Mar. Thou art a scholar; speak to it, Horatio.
        Ber. Looks it not like the King? Mark it, Horatio.
        Hor. Most like. It harrows me with fear and wonder.
        Ber. It would be spoke to.
        Mar. Question it, Horatio.
        Hor. What art thou that usurp'st this time of night
            Together with that fair and warlike form
            In which the majesty of buried Denmark
            Did sometimes march? By heaven I charge thee speak!
        Mar. It is offended.
        Ber. See, it stalks away!
        Hor. Stay! Speak, speak! I charge thee speak!
                                                            Exit Ghost.
        Mar. 'Tis gone and will not answer.
        Ber. How now, Horatio? You tremble and look pale.
            Is not this something more than fantasy?
            What think you on't?
        Hor. Before my God, I might not this believe
            Without the sensible and true avouch
            Of mine own eyes.
        Mar. Is it not like the King?
        Hor. As thou art to thyself.
            Such was the very armour he had on
            When he th' ambitious Norway combated.
            So frown'd he once when, in an angry parle,
            He smote the sledded Polacks on the ice.
            'Tis strange.
        Mar. Thus twice before, and jump at this dead hour,
            With martial stalk hath he gone by our watch.
        Hor. In what particular thought to work I know not;
            But, in the gross and scope of my opinion,
            This bodes some strange eruption to our state.
        Mar. Good now, sit down, and tell me he that knows,
            Why this same strict and most observant watch
            So nightly toils the subject of the land,
            And why such daily cast of brazen cannon
            And foreign mart for implements of war;
            Why such impress of shipwrights, whose sore task
            Does not divide the Sunday from the week.
            What might be toward, that this sweaty haste
            Doth make the night joint-labourer with the day?
            Who is't that can inform me?";

    const TEST_PARAGRAPH2: &str = "Towards the end of November, during a thaw, at nine o’clock one morning, a train on the Warsaw and Petersburg railway was approaching the latter city at full speed.
The morning was so damp and misty that it was only with great difficulty that the day succeeded in breaking;
and it was impossible to distinguish anything more than a few yards away from the carriage windows.
Some of the passengers by this particular train were returning from abroad; but the third-class carriages were the best filled, chiefly with insignificant persons of various occupations and degrees,
picked up at the different stations nearer town.
All of them seemed weary, and most of them had sleepy eyes and a shivering expression, while their complexions generally appeared to have taken on the colour of the fog outside.
When day dawned, two passengers in one of the third-class carriages found themselves opposite each other. Both were young fellows, both were rather poorly dressed, both had remarkable faces,
and both were evidently anxious to start a conversation.
If they had but known why, at this particular moment, they were both remarkable persons, they would undoubtedly have wondered at the strange chance which had set them down opposite to one another in a third-class carriage of the Warsaw Railway Company.
One of them was a young fellow of about twenty-seven, not tall, with black curling hair, and small, grey, fiery eyes. His nose was broad and flat, and he had high cheek bones; his thin lips were constantly compressed into an impudent,
ironical—it might almost be called a malicious—smile;
but his forehead was high and well formed, and atoned for a good deal of the ugliness of the lower part of his face.
A special feature of this physiognomy was its death-like pallor, which gave to the whole man an indescribably emaciated appearance in spite of his hard look,
and at the same time a sort of passionate and suffering expression which did not harmonize with his impudent,
sarcastic smile and keen, self-satisfied bearing.
He wore a large fur—or rather astrachan—overcoat, which had kept him warm all night, while his neighbour had been obliged to bear the full severity of a Russian November night entirely unprepared.
His wide sleeveless mantle with a large cape to it—the sort of cloak one sees upon travellers during the winter months in Switzerland or North Italy—was by no means adapted to the long cold journey through Russia, from Eydkuhnen to St. Petersburg.
The wearer of this cloak was a young fellow, also of about twenty-six or twenty-seven years of age, slightly above the middle height, very fair, with a thin, pointed and very light coloured beard;
his eyes were large and blue, and had an intent look about them, yet that heavy expression which some people affirm to be a peculiarity as well as evidence, of an epileptic subject.
His face was decidedly a pleasant one for all that; refined, but quite colourless, except for the circumstance that at this moment it was blue with cold.
He held a bundle made up of an old faded silk handkerchief that apparently contained all his travelling wardrobe, and wore thick shoes and gaiters, his whole appearance being very un-Russian.
His black-haired neighbour inspected these peculiarities, having nothing better to do, and at length remarked, with that rude enjoyment of the discomforts of others which the common classes so often show:
“Cold?”
“Very,” said his neighbour, readily, “and this is a thaw, too. Fancy if it had been a hard frost! I never thought it would be so cold in the old country. I’ve grown quite out of the way of it.”
“What, been abroad, I suppose?”
“Yes, straight from Switzerland.”
“Wheugh! my goodness!” The black-haired young fellow whistled, and then laughed.
The conversation proceeded. The readiness of the fair-haired young man in the cloak to answer all his opposite neighbour’s questions was surprising.
He seemed to have no suspicion of any impertinence or inappropriateness in the fact of such questions being put to him.
Replying to them, he made known to the inquirer that he certainly had been long absent from Russia, more than four years; that he had been sent abroad for his health;
that he had suffered from some strange nervous malady—a kind of epilepsy, with convulsive spasms. His interlocutor burst out laughing several times at his answers; and more than ever, when to the question, “whether he had been cured?” the patient replied:
“No, they did not cure me.”
“Hey! that’s it! You stumped up your money for nothing, and we believe in those fellows, here!” remarked the black-haired individual, sarcastically.";

    const TEST_PARAGRAPH3: &str = "When the widow hurried away to Pavlofsk, she went straight to Daria Alexeyevna’s house, and telling all she knew, threw her into a state of great alarm.
Both ladies decided to communicate at once with Lebedeff, who, as the friend and landlord of the prince, was also much agitated.
Vera Lebedeff told all she knew, and by Lebedeff’s advice it was decided that all three should go to Petersburg as quickly as possible, in order to avert “what might so easily happen.”
This is how it came about that at eleven o’clock next morning Rogojin’s flat was opened by the police in the presence of Lebedeff, the two ladies, and Rogojin’s own brother, who lived in the wing.
The evidence of the porter went further than anything else towards the success of Lebedeff in gaining the assistance of the police.
He declared that he had seen Rogojin return to the house last night, accompanied by a friend, and that both had gone upstairs very secretly and cautiously.
After this there was no hesitation about breaking open the door, since it could not be got open in any other way.
Rogojin suffered from brain fever for two months. When he recovered from the attack he was at once brought up on trial for murder.
He gave full, satisfactory, and direct evidence on every point; and the prince’s name was, thanks to this, not brought into the proceedings.
Rogojin was very quiet during the progress of the trial. He did not contradict his clever and eloquent counsel, who argued that the brain fever,
or inflammation of the brain, was the cause of the crime; clearly proving that this malady had existed long before the murder was perpetrated, and had been brought on by the sufferings of the accused.
But Rogojin added no words of his own in confirmation of this view, and as before, he recounted with marvellous exactness the details of his crime.
He was convicted, but with extenuating circumstances, and condemned to hard labour in Siberia for fifteen years. He heard his sentence grimly, silently, and thoughtfully. His colossal fortune,
with the exception of the comparatively small portion wasted in the first wanton period of his inheritance, went to his brother, to the great satisfaction of the latter.
The old lady, Rogojin’s mother, is still alive, and remembers her favourite son Parfen sometimes, but not clearly. God spared her the knowledge of this dreadful calamity which had overtaken her house.
Lebedeff, Keller, Gania, Ptitsin, and many other friends of ours continue to live as before. There is scarcely any change in them, so that there is no need to tell of their subsequent doings.
Hippolyte died in great agitation, and rather sooner than he expected, about a fortnight after Nastasia Philipovna’s death. Colia was much affected by these events,
and drew nearer to his mother in heart and sympathy. Nina Alexandrovna is anxious, because he is “thoughtful beyond his years,” but he will, we think, make a useful and active man.
The prince’s further fate was more or less decided by Colia, who selected, out of all the persons he had met during the last six or seven months, Evgenie Pavlovitch, as friend and confidant.
To him he made over all that he knew as to the events above recorded, and as to the present condition of the prince. He was not far wrong in his choice.
Evgenie Pavlovitch took the deepest interest in the fate of the unfortunate “idiot,” and, thanks to his influence, the prince found himself once more with Dr. Schneider, in Switzerland.
Evgenie Pavlovitch, who went abroad at this time, intending to live a long while on the continent, being, as he often said, quite superfluous in Russia, visits his sick friend at Schneider’s every few months.
But Dr. Schneider frowns ever more and more and shakes his head; he hints that the brain is fatally injured; he does not as yet declare that his patient is incurable, but he allows himself to express the gravest fears.
Evgenie takes this much to heart, and he has a heart, as is proved by the fact that he receives and even answers letters from Colia. But besides this,
another trait in his character has become apparent, and as it is a good trait we will make haste to reveal it.
After each visit to Schneider’s establishment, Evgenie Pavlovitch writes another letter, besides that to Colia, giving the most minute particulars concerning the invalid’s condition.
In these letters is to be detected, and in each one more than the last, a growing feeling of friendship and sympathy.
The individual who corresponds thus with Evgenie Pavlovitch, and who engages so much of his attention and respect, is Vera Lebedeff.
We have never been able to discover clearly how such relations sprang up.
Of course the root of them was in the events which we have already recorded, and which so filled Vera with grief on the prince’s account that she fell seriously ill.
But exactly how the acquaintance and friendship came about, we cannot say.";

    /// Always selects the empty first row for the power-of-two offset count
    /// below, and fails promptly if the sampler exceeds its random-draw bound.
    struct EmptyDraws {
        draws: usize,
    }

    impl rand::RngCore for EmptyDraws {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }

        fn next_u64(&mut self) -> u64 {
            self.draws += 1;
            assert!(self.draws <= FSST_SAMPLETARGET);
            u64::MAX
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for chunk in dest.chunks_mut(8) {
                let bytes = self.next_u64().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
        }
    }

    #[test]
    fn test_sample_preserves_small_and_empty_inputs() {
        for input in [Vec::new(), vec![b'x'; FSST_SAMPLETARGET]] {
            let offsets = [0_i32, 0, input.len() as i32, input.len() as i32];
            let mut rng = EmptyDraws { draws: 0 };
            let (sample, sample_offsets) = make_sample(&input, &offsets, &mut rng);
            assert_eq!(sample, input);
            assert_eq!(sample_offsets, offsets);
            assert_eq!(rng.draws, 0);
        }
    }

    #[test]
    fn test_sparse_sample_is_bounded_and_roundtrips() {
        sparse_sample_roundtrip::<i32>();
        sparse_sample_roundtrip::<i64>();
    }

    fn sparse_sample_roundtrip<T: OffsetSizeTrait>() {
        let input = vec![b'x'; FSST_LEAST_INPUT_SIZE];
        // 62 empty values followed by one nonempty value; 64 offsets make
        // every degenerate random draw above select the empty first value.
        let mut offsets = vec![T::zero(); 64];
        offsets[63] = T::from_usize(input.len()).unwrap();
        let mut rng = EmptyDraws { draws: 0 };
        let (sample, sample_offsets) = make_sample(&input, &offsets, &mut rng);
        assert_eq!(rng.draws, FSST_SAMPLETARGET);
        assert_eq!(sample, input);
        assert_eq!(sample_offsets, [T::zero(), offsets[63], offsets[63]]);
        assert!(sample_offsets.capacity() <= FSST_SAMPLETARGET + 2);

        let mut symbols = [0; FSST_SYMBOL_TABLE_SIZE];
        let mut compressed = vec![0; input.len()];
        let mut compressed_offsets = vec![T::zero(); offsets.len()];
        compress(
            &mut symbols,
            &input,
            &offsets,
            &mut compressed,
            &mut compressed_offsets,
        )
        .unwrap();
        let mut decoded = vec![0; compressed.len() * 8];
        let mut decoded_offsets = vec![T::zero(); offsets.len()];
        decompress(
            &symbols,
            &compressed,
            &compressed_offsets,
            &mut decoded,
            &mut decoded_offsets,
        )
        .unwrap();
        assert_eq!(decoded_offsets, offsets);
        assert_eq!(&decoded[..input.len()], input);
    }

    #[test_log::test(tokio::test)]
    async fn test_symbol_new() {
        let st = SymbolTable::new();
        assert!(st.n_symbols == 0);
        for i in 0..=255_u8 {
            assert!(st.symbols[i as usize] == Symbol::from_char(i, i as u16));
        }
        let s = Symbol::from_char(1, 1);
        assert!(s == st.symbols[1]);
        for i in 0..FSST_HASH_TAB_SIZE {
            assert!(st.hash_tab[i] == Symbol::new());
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_fsst() {
        let test_input_size = 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH.len();
        let test_input = TEST_PARAGRAPH.repeat(repeat_num);
        helper(&test_input);

        let test_input_size = 2 * 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH.len();
        let test_input = TEST_PARAGRAPH.repeat(repeat_num);
        helper(&test_input);

        let test_input_size = 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH2.len();
        let test_input = TEST_PARAGRAPH.repeat(repeat_num);
        helper(&test_input);

        let test_input_size = 2 * 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH2.len();
        let test_input = TEST_PARAGRAPH2.repeat(repeat_num);
        helper(&test_input);

        let test_input_size = 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH3.len();
        let test_input = TEST_PARAGRAPH3.repeat(repeat_num); // Also corrected `repea_num` to `repeat_num`
        helper(&test_input);

        let test_input_size = 2 * 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH3.len();
        let test_input = TEST_PARAGRAPH3.repeat(repeat_num); // Also corrected `repea_num` to `repeat_num`
        helper(&test_input);
    }

    fn helper(test_input: &str) {
        let lines_vec = test_input.lines().collect::<Vec<&str>>();
        let string_array = StringArray::from(lines_vec);
        let mut compress_output_buf: Vec<u8> = vec![0; string_array.value_data().len()];
        let mut compress_offset_buf: Vec<i32> = vec![0; string_array.value_offsets().len()];
        let mut symbol_table = [0; FSST_SYMBOL_TABLE_SIZE];
        compress(
            symbol_table.as_mut(),
            string_array.value_data(),
            string_array.value_offsets(),
            &mut compress_output_buf,
            &mut compress_offset_buf,
        )
        .unwrap();
        let mut decompress_output: Vec<u8> = vec![0; compress_output_buf.len() * 8];
        let mut decompress_offsets: Vec<i32> = vec![0; compress_offset_buf.len()];
        decompress(
            &symbol_table,
            &compress_output_buf,
            &compress_offset_buf,
            &mut decompress_output,
            &mut decompress_offsets,
        )
        .unwrap();
        for i in 1..decompress_offsets.len() {
            let s = &decompress_output
                [decompress_offsets[i - 1] as usize..decompress_offsets[i] as usize];
            let original = &string_array.value_data()[string_array.value_offsets().to_vec()[i - 1]
                as usize
                ..string_array.value_offsets().to_vec()[i] as usize];
            assert!(
                s == original,
                "s: {:?}\n\n, original: {:?}",
                std::str::from_utf8(s),
                std::str::from_utf8(original)
            );
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_fsst_64_bit_offsets() {
        let test_input_size = 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH.len();
        let test_input = TEST_PARAGRAPH.repeat(repeat_num);
        helper_64_bit(&test_input);

        let test_input_size = 2 * 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH.len();
        let test_input = TEST_PARAGRAPH.repeat(repeat_num);
        helper_64_bit(&test_input);

        let test_input_size = 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH2.len();
        let test_input = TEST_PARAGRAPH2.repeat(repeat_num);
        helper_64_bit(&test_input);

        let test_input_size = 2 * 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH2.len();
        let test_input = TEST_PARAGRAPH2.repeat(repeat_num);
        helper_64_bit(&test_input);

        let test_input_size = 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH3.len();
        let test_input = TEST_PARAGRAPH3.repeat(repeat_num);
        helper_64_bit(&test_input);

        let test_input_size = 2 * 1024 * 1024;
        let repeat_num = test_input_size / TEST_PARAGRAPH3.len();
        let test_input = TEST_PARAGRAPH3.repeat(repeat_num);
        helper_64_bit(&test_input);
    }

    fn helper_64_bit(test_input: &str) {
        use arrow_array::LargeStringArray;
        let lines_vec = test_input.lines().collect::<Vec<&str>>();
        let string_array = LargeStringArray::from(lines_vec);
        let mut compress_output_buf: Vec<u8> = vec![0; string_array.value_data().len()];
        let mut compress_offset_buf: Vec<i64> = vec![0; string_array.value_offsets().len()];
        let mut symbol_table = [0; FSST_SYMBOL_TABLE_SIZE];
        compress(
            symbol_table.as_mut(),
            string_array.value_data(),
            string_array.value_offsets(),
            &mut compress_output_buf,
            &mut compress_offset_buf,
        )
        .unwrap();
        let mut decompress_output: Vec<u8> = vec![0; compress_output_buf.len() * 8];
        let mut decompress_offsets: Vec<i64> = vec![0; compress_offset_buf.len()];
        decompress(
            &symbol_table,
            &compress_output_buf,
            &compress_offset_buf,
            &mut decompress_output,
            &mut decompress_offsets,
        )
        .unwrap();
        for i in 1..decompress_offsets.len() {
            let s = &decompress_output
                [decompress_offsets[i - 1] as usize..decompress_offsets[i] as usize];
            let original = &string_array.value_data()[string_array.value_offsets().to_vec()[i - 1]
                as usize
                ..string_array.value_offsets().to_vec()[i] as usize];
            assert!(
                s == original,
                "s: {:?}\n\n, original: {:?}",
                std::str::from_utf8(s),
                std::str::from_utf8(original)
            );
        }
    }

    // Build a genuinely FSST-compressed (decoder_switch_on) buffer to exercise the decode-side
    // output-buffer size contract. Returns (symbol_table, compressed_bytes, compressed_offsets).
    fn compress_paragraph() -> ([u8; FSST_SYMBOL_TABLE_SIZE], Vec<u8>, Vec<i32>) {
        let test_input = TEST_PARAGRAPH.repeat((1024 * 1024) / TEST_PARAGRAPH.len());
        let lines_vec = test_input.lines().collect::<Vec<&str>>();
        let string_array = StringArray::from(lines_vec);
        let mut compress_output_buf: Vec<u8> = vec![0; string_array.value_data().len()];
        let mut compress_offset_buf: Vec<i32> = vec![0; string_array.value_offsets().len()];
        let mut symbol_table = [0; FSST_SYMBOL_TABLE_SIZE];
        compress(
            symbol_table.as_mut(),
            string_array.value_data(),
            string_array.value_offsets(),
            &mut compress_output_buf,
            &mut compress_offset_buf,
        )
        .unwrap();
        (symbol_table, compress_output_buf, compress_offset_buf)
    }

    // The decoder writes a full 8-byte word per code, so the output buffer must be at least 8x the
    // compressed input. A buffer sized below 8x must be rejected rather than written out of bounds.
    #[test_log::test(tokio::test)]
    async fn test_decompress_rejects_undersized_output_buffer() {
        let (symbol_table, compressed, compressed_offsets) = compress_paragraph();
        // Sanity check: this input actually engaged FSST compression (decoder_switch_on).
        let st_info = u64::from_ne_bytes(symbol_table[..8].try_into().unwrap());
        assert!(st_info & (1 << 24) != 0, "expected decoder_switch_on input");

        // One byte short of the 8x requirement must be rejected.
        let mut too_small = vec![0u8; compressed.len() * 8 - 1];
        let mut out_offsets = vec![0i32; compressed_offsets.len()];
        let err = decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut too_small,
            &mut out_offsets,
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        // Exactly 8x is the tight bound and must succeed.
        let mut exact = vec![0u8; compressed.len() * 8];
        let mut out_offsets = vec![0i32; compressed_offsets.len()];
        decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut exact,
            &mut out_offsets,
        )
        .unwrap();
    }

    // Only the output's capacity has to cover the 8x bound, so callers need not zero it first
    #[test_log::test(tokio::test)]
    async fn test_decompress_into_uninitialized_capacity() {
        let (symbol_table, compressed, compressed_offsets) = compress_paragraph();
        let mut zeroed = vec![0u8; compressed.len() * 8];
        let mut zeroed_offsets = vec![0i32; compressed_offsets.len()];
        decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut zeroed,
            &mut zeroed_offsets,
        )
        .unwrap();

        // Garbage in the spare capacity must never show up in the output
        let mut out = vec![0xAAu8; compressed.len() * 8];
        out.clear();
        let mut out_offsets = vec![0i32; compressed_offsets.len()];
        decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap();
        assert_eq!(out, zeroed);
        assert_eq!(out_offsets, zeroed_offsets);
    }

    fn declared_lens_range(symbol_table: &[u8]) -> std::ops::Range<usize> {
        let n_symbols = (u64::from_ne_bytes(symbol_table[..8].try_into().unwrap()) & 255) as usize;
        let start = 8 + n_symbols * 8;
        start..start + n_symbols
    }

    #[test_log::test(tokio::test)]
    async fn test_decompress_rejects_corrupt_symbol_length() {
        let (mut symbol_table, compressed, compressed_offsets) = compress_paragraph();
        let lens = declared_lens_range(&symbol_table);
        assert!(!lens.is_empty(), "expected at least one declared symbol");
        symbol_table[lens.start] = 9;

        let mut out = vec![0u8; compressed.len() * 8];
        let mut out_offsets = vec![0i32; compressed_offsets.len()];
        let err = decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("symbol length"),
            "unexpected error: {err}"
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_decompress_rejects_zero_declared_symbol_length() {
        let (mut symbol_table, compressed, compressed_offsets) = compress_paragraph();
        let lens = declared_lens_range(&symbol_table);
        symbol_table[lens.start] = 0;

        let mut out = vec![0u8; compressed.len() * 8];
        let mut out_offsets = vec![0i32; compressed_offsets.len()];
        let err = decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("symbol length"),
            "unexpected error: {err}"
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_decompress_rejects_out_of_range_offset() {
        let (symbol_table, compressed, mut compressed_offsets) = compress_paragraph();
        let last = compressed_offsets.len() - 1;
        compressed_offsets[last] = i32::try_from(compressed.len()).unwrap() + 1;

        let mut out = vec![0u8; compressed.len() * 8];
        let mut out_offsets = vec![0i32; compressed_offsets.len()];
        let err = decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("out of bounds"),
            "unexpected error: {err}"
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_decompress_rejects_decreasing_offset() {
        let (symbol_table, compressed, mut compressed_offsets) = compress_paragraph();
        assert!(compressed_offsets.len() >= 2);
        if compressed_offsets[0] == 0 {
            compressed_offsets[0] = 1;
        }
        compressed_offsets[1] = compressed_offsets[0] - 1;

        let mut out = vec![0u8; compressed.len() * 8];
        let mut out_offsets = vec![0i32; compressed_offsets.len()];
        let err = decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("decreases"),
            "unexpected error: {err}"
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_decompress_rejects_negative_offset() {
        let (symbol_table, compressed, mut compressed_offsets) = compress_paragraph();
        compressed_offsets[0] = -1;

        let mut out = vec![0u8; compressed.len() * 8];
        let mut out_offsets = vec![0i32; compressed_offsets.len()];
        let err = decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("negative") || err.to_string().contains("out of bounds"),
            "unexpected error: {err}"
        );
    }

    #[test_log::test(tokio::test)]
    async fn test_decompress_accepts_max_symbol_length() {
        let (symbol_table, compressed, compressed_offsets) = compress_paragraph();
        let lens = declared_lens_range(&symbol_table);
        assert!(
            symbol_table[lens].contains(&(MAX_SYMBOL_LENGTH as u8)),
            "expected encoder to emit at least one 8-byte symbol"
        );

        let mut out = vec![0u8; compressed.len() * 8];
        let mut out_offsets = vec![0i32; compressed_offsets.len()];
        decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap();
    }

    #[test_log::test(tokio::test)]
    async fn test_undeclared_code_does_not_overflow() {
        let (symbol_table, mut compressed, compressed_offsets) = compress_paragraph();
        let n_symbols = (u64::from_ne_bytes(symbol_table[..8].try_into().unwrap()) & 255) as usize;
        if n_symbols >= 255 {
            return;
        }
        let undeclared = n_symbols as u8;
        if let Some(byte) = compressed.iter_mut().find(|byte| **byte != FSST_ESC) {
            *byte = undeclared;
        }

        let mut out = vec![0u8; compressed.len() * 8];
        let mut out_offsets = vec![0i32; compressed_offsets.len()];
        decompress(
            &symbol_table,
            &compressed,
            &compressed_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap();
    }

    fn switch_off_roundtrip() -> ([u8; FSST_SYMBOL_TABLE_SIZE], Vec<u8>, Vec<i32>) {
        let input = b"raw";
        let offsets = [0_i32, 3];
        let mut table = [0_u8; FSST_SYMBOL_TABLE_SIZE];
        let mut compressed = vec![0; input.len()];
        let mut compressed_offsets = vec![0_i32; offsets.len()];
        compress(
            &mut table,
            input,
            &offsets,
            &mut compressed,
            &mut compressed_offsets,
        )
        .unwrap();
        let st_info = u64::from_ne_bytes(table[..8].try_into().unwrap());
        assert!(st_info & (1 << 24) == 0, "expected decoder_switch_on off");
        (table, compressed, compressed_offsets)
    }

    #[test]
    fn test_decompress_rejects_corrupt_offsets_when_switch_off() {
        let (table, compressed, _) = switch_off_roundtrip();
        let corrupt_offsets = [-1_i32, 99];
        let mut out = vec![0; compressed.len()];
        let mut out_offsets = vec![0_i32; corrupt_offsets.len()];
        let err = decompress(
            &table,
            &compressed,
            &corrupt_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_decompress_switch_off_accepts_valid_offsets() {
        let (table, compressed, compressed_offsets) = switch_off_roundtrip();
        let mut out = vec![0; compressed.len()];
        let mut out_offsets = vec![0_i32; compressed_offsets.len()];
        decompress(
            &table,
            &compressed,
            &compressed_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap();
        assert_eq!(out, compressed);
        assert_eq!(out_offsets, compressed_offsets);
    }

    fn assert_missing_escape_payload(table: &[u8], bytes: &[u8], offsets: &[i32]) {
        let mut out = vec![0_u8; bytes.len() * 8];
        let mut out_offsets = vec![0_i32; offsets.len()];
        let err = decompress(table, bytes, offsets, &mut out, &mut out_offsets).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("escape"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_decompress_rejects_dangling_escape_in_scalar_tail() {
        let (table, _, _) = compress_paragraph();
        assert_missing_escape_payload(&table, &[0, FSST_ESC], &[0, 2]);
        assert_missing_escape_payload(&table, &[FSST_ESC], &[0, 1]);
    }

    #[test]
    fn test_decompress_rejects_dangling_escape_in_fast_path() {
        let (table, _, _) = compress_paragraph();
        // 4-byte window ending in ESC: payload sits at index 4, outside in_end.
        assert_missing_escape_payload(&table, &[0, 0, 0, FSST_ESC], &[0, 4]);
    }

    #[test]
    fn test_decompress_rejects_escape_payload_from_next_value() {
        let (table, _, _) = compress_paragraph();
        // First value ends on ESC; the next value's first byte must not be stolen as payload.
        assert_missing_escape_payload(&table, &[0, 0, 0, FSST_ESC, b'X'], &[0, 4, 5]);
    }

    #[test]
    fn test_decompress_accepts_escape_with_payload() {
        let (table, _, _) = compress_paragraph();
        let bytes = [FSST_ESC, b'A'];
        let offsets = [0_i32, 2];
        let mut out = vec![0_u8; bytes.len() * 8];
        let mut out_offsets = vec![0_i32; offsets.len()];
        decompress(&table, &bytes, &offsets, &mut out, &mut out_offsets).unwrap();
        assert_eq!(&out[..], b"A");
        assert_eq!(out_offsets, [0, 1]);
    }

    #[test]
    fn test_decompress_accepts_fast_path_escape_with_payload() {
        let (table, _, _) = compress_paragraph();
        let bytes = [0, 0, 0, FSST_ESC, b'Z'];
        let offsets = [0_i32, 5];
        let mut out = vec![0_u8; bytes.len() * 8];
        let mut out_offsets = vec![0_i32; offsets.len()];
        decompress(&table, &bytes, &offsets, &mut out, &mut out_offsets).unwrap();
        assert_eq!(out_offsets[0], 0);
        assert_eq!(*out.last().unwrap(), b'Z');
        assert_eq!(out_offsets[1], i32::try_from(out.len()).unwrap());
    }

    #[test]
    fn test_boundary_string_lengths_and_reused_decompressor() {
        let mut strings: Vec<String> = Vec::new();
        // Exercise every string length 0..=24 (all 8-byte SWAR / 4-byte SWAR / scalar tail boundaries)
        // and 510..=514 (former 511-byte chunk boundary), plus > 32 KiB total to trigger active FSST.
        for rep in 0..120 {
            for len in (0..=24).chain(510..=514) {
                let s: String = (0..len)
                    .map(|idx| {
                        let b = b'a' + ((idx + rep) % 26) as u8;
                        b as char
                    })
                    .collect();
                strings.push(s);
            }
        }
        let arr = StringArray::from(strings.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        let mut symbol_table = [0u8; FSST_SYMBOL_TABLE_SIZE];
        let (compressed, compressed_offsets) =
            compress_into(&mut symbol_table, arr.value_data(), arr.value_offsets()).unwrap();

        let decoder = FsstDecompressor::try_new(&symbol_table).unwrap();
        assert!(decoder.decoder_switch_on);

        // Reuse the compiled decoder across slices of the compressed stream
        let mid = strings.len() / 2;
        let split_byte = compressed_offsets[mid] as usize;
        let first_offsets = &compressed_offsets[..=mid];
        let second_offsets: Vec<i32> = compressed_offsets[mid..]
            .iter()
            .map(|&off| off - compressed_offsets[mid])
            .collect();

        let (dec1, off1) = decoder
            .decompress_into(&compressed[..split_byte], first_offsets)
            .unwrap();
        let (dec2, off2) = decoder
            .decompress_into(&compressed[split_byte..], &second_offsets)
            .unwrap();

        for i in 0..mid {
            assert_eq!(
                &dec1[off1[i] as usize..off1[i + 1] as usize],
                strings[i].as_bytes()
            );
        }
        for i in 0..(strings.len() - mid) {
            assert_eq!(
                &dec2[off2[i] as usize..off2[i + 1] as usize],
                strings[mid + i].as_bytes()
            );
        }
    }

    #[test]
    fn test_swar8_escape_at_every_lane_position() {
        let (table, _, _) = compress_paragraph();
        let decoder = FsstDecompressor::try_new(&table).unwrap();
        for esc_pos in 0..8usize {
            let mut bytes = vec![0u8; 9];
            bytes[esc_pos] = FSST_ESC;
            bytes[esc_pos + 1] = b'Q';
            // Also test neighbor 0xFE byte right after 0xFF to verify borrow propagation safety
            if esc_pos + 2 < bytes.len() {
                bytes[esc_pos + 2] = FSST_ESC;
                bytes.push(0xFE);
            }
            let offsets = [0_i32, i32::try_from(bytes.len()).unwrap()];
            let (decoded, decoded_offsets) = decoder.decompress_into(&bytes, &offsets).unwrap();
            assert_eq!(decoded_offsets[0], 0);
            assert_eq!(decoded_offsets[1] as usize, decoded.len());
            assert!(decoded.contains(&b'Q'));
        }
        // Dangling escape at lane 7 of an 8-byte window must error cleanly
        let dangling8 = [0u8, 0, 0, 0, 0, 0, 0, FSST_ESC];
        assert_missing_escape_payload(&table, &dangling8, &[0, 8]);
    }

    #[test]
    fn test_compress_rejects_invalid_offsets_and_handles_zero_values() {
        let input = vec![b'a'; FSST_LEAST_INPUT_SIZE + 64];
        let mut symbol_table = [0u8; FSST_SYMBOL_TABLE_SIZE];

        // Out-of-range offset beyond `input.len()` (including after >16 KiB sampled prefix)
        let mut oob_offsets: Vec<i32> = (0..=512).map(|i| i * 64).collect();
        *oob_offsets.last_mut().unwrap() = (input.len() + 8) as i32;
        let err = compress_into(&mut symbol_table, &input, &oob_offsets).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

        // Decreasing offsets (would otherwise compress overlapping windows past `2 * input.len()`)
        let decreasing_offsets = [0_i32, input.len() as i32, 0, input.len() as i32];
        let mut out = vec![0u8; input.len()];
        let mut out_offsets = vec![0i32; decreasing_offsets.len()];
        let err = compress(
            &mut symbol_table,
            &input,
            &decreasing_offsets,
            &mut out,
            &mut out_offsets,
        )
        .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

        // Negative offset
        let neg_offsets = [-1_i32, 16];
        let err = compress_into(&mut symbol_table, &input, &neg_offsets).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

        // Zero strings on a large shared backing buffer must not underflow/divide-by-zero in sampling
        for empty_offsets in [&[][..], &[0_i32][..], &[128_i32][..]] {
            let (comp, comp_off) = compress_into(&mut symbol_table, &input, empty_offsets).unwrap();
            assert!(comp.is_empty());
            assert_eq!(comp_off, vec![0_i32; empty_offsets.len()]);
            let dec = FsstDecompressor::try_new(&symbol_table).unwrap();
            let (rt, rt_off) = dec.decompress_into(&comp, &comp_off).unwrap();
            assert!(rt.is_empty());
            assert_eq!(rt_off, comp_off);
        }
    }
}
