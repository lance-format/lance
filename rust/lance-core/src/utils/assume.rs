// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Assert an invariant that should also be visible to the optimizer.
///
/// Unlike [`debug_assert!`], this remains checked in release builds. This is
/// required because the macro can be invoked from safe Rust and an invalid
/// assumption must not become undefined behavior.
#[macro_export]
macro_rules! assume {
    ($cond:expr) => {
        assert!($cond)
    };
    ($cond:expr, $($arg:tt)+) => {
        assert!($cond, $($arg)+)
    };
}

/// Helper macro for equality assumptions.
#[macro_export]
macro_rules! assume_eq {
    ($left:expr, $right:expr) => {
        assert_eq!($left, $right)
    };
    ($left:expr, $right:expr, $($arg:tt)+) => {
        assert_eq!($left, $right, $($arg)+)
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn assume_rejects_false_conditions() {
        assert!(std::panic::catch_unwind(|| assume!(false)).is_err());
        assert!(std::panic::catch_unwind(|| assume!(false, "invalid condition")).is_err());
    }

    #[test]
    fn assume_eq_rejects_unequal_values() {
        assert!(std::panic::catch_unwind(|| assume_eq!(1, 2)).is_err());
        assert!(std::panic::catch_unwind(|| assume_eq!(1, 2, "invalid equality")).is_err());
    }
}
