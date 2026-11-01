// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

// JSON text constants
pub const UNICODE_LEN: usize = 4;

// JSON text escape characters constants
pub const BS: char = '\x5C'; // \\ Backslash
pub const QU: char = '\x22'; // \" Double quotation mark
pub const SD: char = '\x2F'; // \/ Slash or divide
pub const BB: char = '\x08'; // \b Backspace
pub const FF: char = '\x0C'; // \f Formfeed Page Break
pub const NN: char = '\x0A'; // \n Newline
pub const RR: char = '\x0D'; // \r Carriage Return
pub const TT: char = '\x09'; // \t Horizontal Tab

pub const MAX_DECIMAL128_PRECISION: usize = 38;
pub const MAX_DECIMAL256_PRECISION: usize = 76;

pub const UINT64_MIN: i128 = 0i128;
pub const UINT64_MAX: i128 = 18_446_744_073_709_551_615i128;
pub const INT64_MIN: i128 = -9_223_372_036_854_775_808i128;
pub const INT64_MAX: i128 = 9_223_372_036_854_775_807i128;
