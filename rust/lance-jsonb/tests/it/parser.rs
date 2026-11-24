// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::borrow::Cow;

use lance_jsonb::{Number, Object, Value, parse_value};

fn test_parse_err(errors: &[(&str, &'static str)]) {
    for &(s, err) in errors {
        let res = parse_value(s.as_bytes());
        assert!(res.is_err());
        assert_eq!(res.err().unwrap().to_string(), err);
    }
}

fn test_parse_ok(tests: Vec<(&str, Value<'_>)>) {
    for (s, val) in tests {
        assert_eq!(parse_value(s.as_bytes()).unwrap(), val);
    }
}

#[test]
fn test_parse_null() {
    test_parse_err(&[
        ("n", "EOF while parsing a value, pos 1"),
        ("nul", "expected ident, pos 2"),
        ("nulla", "trailing characters, pos 5"),
    ]);

    test_parse_ok(vec![("null", Value::Null)]);
    // Extended JSON null syntax, allow uppercase letters and empty string is treated as NULL
    test_parse_ok(vec![
        ("  ", Value::Null),
        ("NULL", Value::Null),
        ("Null", Value::Null),
    ]);
}

#[test]
fn test_parse_boolean() {
    test_parse_err(&[
        ("t", "EOF while parsing a value, pos 1"),
        ("truz", "expected ident, pos 4"),
        ("f", "EOF while parsing a value, pos 1"),
        ("faz", "expected ident, pos 3"),
        ("truea", "trailing characters, pos 5"),
        ("falsea", "trailing characters, pos 6"),
    ]);

    test_parse_ok(vec![
        ("true", Value::Bool(true)),
        (" true ", Value::Bool(true)),
        ("false", Value::Bool(false)),
        (" false ", Value::Bool(false)),
    ]);
    // Extended JSON boolean syntax, allow uppercase letters
    test_parse_ok(vec![
        ("TRUE", Value::Bool(true)),
        (" True ", Value::Bool(true)),
        ("FALSE", Value::Bool(false)),
        (" falSE ", Value::Bool(false)),
    ]);
}

#[test]
fn test_parse_number_errors() {
    test_parse_err(&[
        ("+", "invalid number, pos 1"),
        (".", "invalid number, pos 1"),
        ("-", "invalid number, pos 1"),
        ("\\0", "expected value, pos 1"),
        ("1.a", "trailing characters, pos 3"),
        ("1e", "invalid number, pos 2"),
        ("1e+", "invalid number, pos 3"),
        ("1a", "trailing characters, pos 2"),
    ]);
}

#[test]
fn test_parse_i64() {
    let i64_min = i64::MIN.to_string();
    let i64_max = i64::MAX.to_string();
    let tests = vec![
        ("-2", Value::Number(Number::Int64(-2))),
        ("-1234", Value::Number(Number::Int64(-1234))),
        (" -1234 ", Value::Number(Number::Int64(-1234))),
        (&i64_min, Value::Number(Number::Int64(i64::MIN))),
        (&i64_max, Value::Number(Number::UInt64(i64::MAX as u64))),
    ];
    test_parse_ok(tests);
}

#[test]
fn test_parse_u64() {
    let u64_max = u64::MAX.to_string();
    let tests = vec![
        ("0", Value::Number(Number::UInt64(0u64))),
        ("3", Value::Number(Number::UInt64(3u64))),
        ("1234", Value::Number(Number::UInt64(1234))),
        (&u64_max, Value::Number(Number::UInt64(u64::MAX))),
    ];
    test_parse_ok(tests);
}

#[test]
fn test_parse_f64() {
    let i64_min_minus_one = format!("{}", (i64::MIN as f64) - 1.0);
    let u64_max_plus_one = format!("{}", (u64::MAX as f64) + 1.0);
    let epsilon = format!("{}", f64::EPSILON);

    let tests = vec![
        (
            "100e777777777777777777777777777",
            Value::Number(Number::Float64(f64::INFINITY)),
        ),
        (
            "-100e777777777777777777777777777",
            Value::Number(Number::Float64(f64::NEG_INFINITY)),
        ),
        (
            "1000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000",
            Value::Number(Number::Float64(f64::INFINITY)),
        ),
        (
            "1000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             .0e9",
            Value::Number(Number::Float64(f64::INFINITY)),
        ),
        (
            "1000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             e9",
            Value::Number(Number::Float64(f64::INFINITY)),
        ),
        ("0.0", Value::Number(Number::Float64(0.0f64))),
        ("3.0", Value::Number(Number::Float64(3.0f64))),
        ("3.1", Value::Number(Number::Float64(3.1f64))),
        ("-1.2", Value::Number(Number::Float64(-1.2f64))),
        ("0.4", Value::Number(Number::Float64(0.4f64))),
        ("0.4", Value::Number(Number::Float64(0.4f64))),
        (
            "2.638344616030823e-256",
            Value::Number(Number::Float64(2.638344616030823e-256)),
        ),
        ("3.00", Value::Number(Number::Float64(3.0f64))),
        ("0.4e5", Value::Number(Number::Float64(0.4e5))),
        ("0.4e+5", Value::Number(Number::Float64(0.4e5))),
        ("0.4e15", Value::Number(Number::Float64(0.4e15))),
        ("0.4e+15", Value::Number(Number::Float64(0.4e15))),
        ("0.4e-01", Value::Number(Number::Float64(0.4e-1))),
        (" 0.4e-01 ", Value::Number(Number::Float64(0.4e-1))),
        ("0.4e-001", Value::Number(Number::Float64(0.4e-1))),
        ("0.4e-0", Value::Number(Number::Float64(0.4e0))),
        ("0.00e00", Value::Number(Number::Float64(0.0))),
        ("0.00e+00", Value::Number(Number::Float64(0.0))),
        ("0.00e-00", Value::Number(Number::Float64(0.0))),
        ("3.5E-2147483647", Value::Number(Number::Float64(0.0))),
        (
            "0.0100000000000000000001",
            Value::Number(Number::Float64(0.01)),
        ),
        (
            &i64_min_minus_one,
            Value::Number(Number::Float64((i64::MIN as f64) - 1.0)),
        ),
        (
            &u64_max_plus_one,
            Value::Number(Number::Float64((u64::MAX as f64) + 1.0)),
        ),
        (&epsilon, Value::Number(Number::Float64(f64::EPSILON))),
        (
            "0.0000000000000000000000000000000000000000000000000123e50",
            Value::Number(Number::Float64(1.23)),
        ),
        (
            "100e-777777777777777777777777777",
            Value::Number(Number::Float64(0.0)),
        ),
        (
            "1010101010101010101010101010101010101010",
            Value::Number(Number::Float64(1.010_101_010_101_01e39)),
        ),
        (
            "0.1010101010101010101010101010101010101010",
            Value::Number(Number::Float64(0.101_010_101_010_101_01)),
        ),
        (
            "0e1000000000000000000000000000000000000000000000",
            Value::Number(Number::Float64(0.0)),
        ),
        (
            "1000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             00000000",
            Value::Number(Number::Float64(1e308)),
        ),
        (
            "1000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             .0e8",
            Value::Number(Number::Float64(1e308)),
        ),
        (
            "1000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             e8",
            Value::Number(Number::Float64(1e308)),
        ),
        (
            "1000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000000000000000000000000000000000000000000000\
             000000000000000000e-10",
            Value::Number(Number::Float64(1e308)),
        ),
    ];
    test_parse_ok(tests);

    // Extended JSON number syntax
    let extended_tests = vec![
        ("+1", Value::Number(Number::Int64(1))),
        ("00", Value::Number(Number::UInt64(0))),
        (".0", Value::Number(Number::UInt64(0))),
        ("0.", Value::Number(Number::UInt64(0))),
        ("1.", Value::Number(Number::UInt64(1))),
        ("1.e1", Value::Number(Number::Float64(10.0))),
        ("nan", Value::Number(Number::Float64(f64::NAN))),
        ("+infinity", Value::Number(Number::Float64(f64::INFINITY))),
        ("INFINITY", Value::Number(Number::Float64(f64::INFINITY))),
        (
            "-INFINITY",
            Value::Number(Number::Float64(f64::NEG_INFINITY)),
        ),
        ("0xdecaf", Value::Number(Number::UInt64(912559))),
        (
            "0xdecaf.124",
            Value::Number(Number::Float64(912559.0712890625)),
        ),
    ];
    test_parse_ok(extended_tests);
}

#[test]
fn test_parse_string() {
    test_parse_err(&[
        ("\"", "EOF while parsing a value, pos 1"),
        ("\"lol", "EOF while parsing a value, pos 4"),
        ("\"lol\"a", "trailing characters, pos 6"),
    ]);

    test_parse_ok(vec![
        ("\"abc\"", Value::String(Cow::from("abc"))),
        ("\"n\"", Value::String(Cow::from("n"))),
        ("\"\\\"\"", Value::String(Cow::from("\""))),
        ("\"\\\\\"", Value::String(Cow::from("\\"))),
        ("\"/\"", Value::String(Cow::from("/"))),
        ("\"\\b\"", Value::String(Cow::from("\x08"))),
        ("\"\\f\"", Value::String(Cow::from("\x0c"))),
        ("\"\\n\"", Value::String(Cow::from("\n"))),
        ("\"\\r\"", Value::String(Cow::from("\r"))),
        ("\"\\t\"", Value::String(Cow::from("\t"))),
        ("\"\\u000b\"", Value::String(Cow::from("\x0B"))),
        ("\"\\u000B\"", Value::String(Cow::from("\x0B"))),
        ("\"\u{3A3}\"", Value::String(Cow::from("\u{3A3}"))),
    ]);

    test_parse_ok(vec![
        ("\"\"", Value::String(Cow::from(""))),
        ("\"foo\"", Value::String(Cow::from("foo"))),
        (" \"foo\" ", Value::String(Cow::from("foo"))),
        ("\"\\\"\"", Value::String(Cow::from("\""))),
        ("\"\\b\"", Value::String(Cow::from("\x08"))),
        ("\"\\n\"", Value::String(Cow::from("\n"))),
        ("\"\\r\"", Value::String(Cow::from("\r"))),
        ("\"\\t\"", Value::String(Cow::from("\t"))),
        ("\"\\u12ab\"", Value::String(Cow::from("\u{12ab}"))),
        ("\"\\uAB12\"", Value::String(Cow::from("\u{AB12}"))),
        ("\"\\uD83C\\uDF95\"", Value::String(Cow::from("\u{1F395}"))),
        (r#""\u5b57""#, Value::String(Cow::from("字"))),
        (r#""\u0000""#, Value::String(Cow::from("\0"))),
        (r#""\uDEAD""#, Value::String(Cow::from("\\uDEAD"))),
        (
            r#""\uDC00\uD800""#,
            Value::String(Cow::from("\\uDC00\\uD800")),
        ),
        (
            r#""\uD800\uDA00""#,
            Value::String(Cow::from("\\uD800\\uDA00")),
        ),
        (r#""\uD803\uDC0B""#, Value::String(Cow::from("𐰋"))),
        (r#""\uD83D\uDC8E""#, Value::String(Cow::from("💎"))),
        (
            r#""\\\uD83D\\\uDC8E""#,
            Value::String(Cow::from("\\\\uD83D\\\\uDC8E")),
        ),
        (
            r#""\"ab\"\uD803\uDC0B测试""#,
            Value::String(Cow::from("\"ab\"𐰋测试")),
        ),
        (r#""⚠\u{fe0f}""#, Value::String(Cow::from("⚠\u{fe0f}"))),
    ]);

    // Extended JSON string syntax
    let extended_tests = vec![("'abcd'", Value::String(Cow::from("abcd")))];
    test_parse_ok(extended_tests);
}

#[test]
fn test_parse_array() {
    test_parse_err(&[
        ("[", "EOF while parsing a value, pos 1"),
        ("[ ", "EOF while parsing a value, pos 2"),
        ("[1", "EOF while parsing a value, pos 2"),
        ("[1,", "EOF while parsing a value, pos 3"),
        ("[1 2]", "expected `,` or `]`, pos 3"),
        ("[]a", "trailing characters, pos 3"),
    ]);

    let tests = vec![
        ("[]", Value::Array(vec![])),
        ("[ ]", Value::Array(vec![])),
        ("[null]", Value::Array(vec![Value::Null])),
        (" [ null ]", Value::Array(vec![Value::Null])),
        ("[true]", Value::Array(vec![Value::Bool(true)])),
        (
            "[3,1]",
            Value::Array(vec![
                Value::Number(Number::UInt64(3)),
                Value::Number(Number::UInt64(1)),
            ]),
        ),
        (
            "[[3], [1, 2]]",
            Value::Array(vec![
                Value::Array(vec![Value::Number(Number::UInt64(3))]),
                Value::Array(vec![
                    Value::Number(Number::UInt64(1)),
                    Value::Number(Number::UInt64(2)),
                ]),
            ]),
        ),
        ("[1]", Value::Array(vec![Value::Number(Number::UInt64(1))])),
        (
            "[1, 2]",
            Value::Array(vec![
                Value::Number(Number::UInt64(1)),
                Value::Number(Number::UInt64(2)),
            ]),
        ),
        (
            "[1, 2, 3]",
            Value::Array(vec![
                Value::Number(Number::UInt64(1)),
                Value::Number(Number::UInt64(2)),
                Value::Number(Number::UInt64(3)),
            ]),
        ),
        (
            "[1, [2, 3]]",
            Value::Array(vec![
                Value::Number(Number::UInt64(1)),
                Value::Array(vec![
                    Value::Number(Number::UInt64(2)),
                    Value::Number(Number::UInt64(3)),
                ]),
            ]),
        ),
    ];

    test_parse_ok(tests);

    // Extended JSON array syntax
    let extended_tests = vec![
        (
            "[1, ]",
            Value::Array(vec![Value::Number(Number::UInt64(1)), Value::Null]),
        ),
        (
            "[ , 2, 3]",
            Value::Array(vec![
                Value::Null,
                Value::Number(Number::UInt64(2)),
                Value::Number(Number::UInt64(3)),
            ]),
        ),
        ("[ , ]", Value::Array(vec![Value::Null, Value::Null])),
    ];
    test_parse_ok(extended_tests);
}

#[test]
fn test_parse_object() {
    test_parse_err(&[
        ("{", "EOF while parsing a value, pos 1"),
        ("{ ", "EOF while parsing a value, pos 2"),
        ("{1", "object attribute name cannot be a number, pos 2"),
        ("{ \"a\"", "EOF while parsing a value, pos 5"),
        ("{\"a\"", "EOF while parsing a value, pos 4"),
        ("{\"a\" ", "EOF while parsing a value, pos 5"),
        ("{\"a\" 1", "expected `:`, pos 5"),
        ("{\"a\":", "EOF while parsing a value, pos 5"),
        ("{\"a\":1", "EOF while parsing a value, pos 6"),
        ("{\"a\":1 1", "expected `,` or `}`, pos 7"),
        ("{\"a\":1,", "EOF while parsing a value, pos 7"),
        ("{}a", "trailing characters, pos 3"),
        (
            "{\"k\":\"v\",\"k\":\"v2\"}",
            "duplicate object attribute \"k\", pos 12",
        ),
    ]);

    let mut obj1 = Object::new();
    obj1.insert("a".to_string(), Value::Number(Number::UInt64(3)));
    let mut obj2 = Object::new();
    obj2.insert("a".to_string(), Value::Number(Number::UInt64(3)));
    obj2.insert("b".to_string(), Value::Number(Number::UInt64(4)));
    let mut obj3 = Object::new();
    let mut obj3val = Object::new();
    obj3val.insert("b".to_string(), Value::Number(Number::UInt64(3)));
    obj3val.insert("c".to_string(), Value::Number(Number::UInt64(4)));
    obj3.insert("a".to_string(), Value::Object(obj3val));
    let mut obj4 = Object::new();
    obj4.insert("c".to_string(), Value::Null);
    let mut obj5 = Object::new();
    obj5.insert("d".to_string(), Value::Number(Number::UInt64(5)));
    let mut obj6 = Object::new();
    obj6.insert("_test123中文".to_string(), Value::Number(Number::UInt64(6)));

    test_parse_ok(vec![
        (r#"{}"#, Value::Object(Object::new())),
        (r#"{ }"#, Value::Object(Object::new())),
        (r#"{"a":3}"#, Value::Object(obj1.clone())),
        (r#"{ "a" : 3 }"#, Value::Object(obj1.clone())),
        (r#"{"a":3,"b":4}"#, Value::Object(obj2.clone())),
        (r#" { "a" : 3 , "b" : 4 } "#, Value::Object(obj2)),
        (r#"{"a": {"b": 3, "c": 4}}"#, Value::Object(obj3.clone())),
        (r#"{"c":null}"#, Value::Object(obj4)),
        (r#"{\t\n\r "d":  5}"#, Value::Object(obj5.clone())),
        (r#"{ \x0C "d":  5}"#, Value::Object(obj5.clone())),
    ]);

    // Extended JSON string syntax
    let extended_tests = vec![
        ("{'a':3}", Value::Object(obj1)),
        ("{a:{b:3, c:4}}", Value::Object(obj3)),
        ("{d:5}", Value::Object(obj5)),
        ("{_test123中文 :6}", Value::Object(obj6)),
    ];
    test_parse_ok(extended_tests);
}
