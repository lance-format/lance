// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Parse a string into a boolean value.
pub fn str_is_truthy(val: &str) -> bool {
    val.eq_ignore_ascii_case("1")
        | val.eq_ignore_ascii_case("true")
        | val.eq_ignore_ascii_case("on")
        | val.eq_ignore_ascii_case("yes")
        | val.eq_ignore_ascii_case("y")
}

/// Parse a string into an optional boolean value.
///
/// Returns `Some(true)` for truthy values (1/true/on/yes/y, case-insensitive).
/// Returns `Some(false)` for falsy values (0/false/off/no/n, case-insensitive).
/// Returns `None` for unrecognized values.
pub fn str_to_bool(val: &str) -> Option<bool> {
    if str_is_truthy(val) {
        Some(true)
    } else if val.eq_ignore_ascii_case("0")
        || val.eq_ignore_ascii_case("false")
        || val.eq_ignore_ascii_case("off")
        || val.eq_ignore_ascii_case("no")
        || val.eq_ignore_ascii_case("n")
    {
        Some(false)
    } else {
        None
    }
}

/// Parse an environment variable as a truthy-only boolean.
///
/// Returns `default_value` if the env var is not set.
/// Returns `true` only for truthy values (1/true/on/yes/y, case-insensitive).
/// Returns `false` for all other set values.
pub fn parse_env_as_bool(env_var_name: &str, default_value: bool) -> bool {
    std::env::var(env_var_name)
        .ok()
        .map(|value| str_is_truthy(value.trim()))
        .unwrap_or(default_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_str_to_bool_truthy() {
        for val in [
            "1", "true", "True", "TRUE", "on", "ON", "yes", "YES", "y", "Y",
        ] {
            assert_eq!(
                str_to_bool(val),
                Some(true),
                "expected Some(true) for {:?}",
                val
            );
        }
    }

    #[test]
    fn test_str_to_bool_falsy() {
        for val in [
            "0", "false", "False", "FALSE", "off", "OFF", "no", "NO", "n", "N",
        ] {
            assert_eq!(
                str_to_bool(val),
                Some(false),
                "expected Some(false) for {:?}",
                val
            );
        }
    }

    #[test]
    fn test_str_to_bool_unknown() {
        for val in ["", "2", "maybe", "truthy", "nonsense"] {
            assert_eq!(str_to_bool(val), None, "expected None for {:?}", val);
        }
    }
}
