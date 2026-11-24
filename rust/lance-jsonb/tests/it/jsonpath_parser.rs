// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::io::Write;

use goldenfile::Mint;
use lance_jsonb::Error;
use lance_jsonb::Number;
use lance_jsonb::jsonpath::parse_json_path;
use lance_jsonb::jsonpath::{Expr, Path, PathValue};
use rstest::rstest;

#[test]
fn test_json_path() {
    let mut mint = Mint::new("tests/it/testdata");
    let mut file = mint.new_goldenfile("json_path.txt").unwrap();
    let cases = &[
        r#"$"#,
        r#"$.*"#,
        r#"$.**"#,
        r#"$.**{2 to last}"#,
        r#"$[*]"#,
        r#"5 + 5"#,
        r#"10 - 5"#,
        r#"10 * 5"#,
        r#"10 / 5"#,
        r#"10 % 5"#,
        r#"$.store.book[*].*"#,
        // r#"$.store.book[*].* + 5"#,
        r#"$.store.book[0].price"#,
        r#"+$.store.book[0].price"#,
        r#"-$.store.book[0].price"#,
        r#"$.store.book[0].price + 5"#,
        r#"$.store.book[last].isbn"#,
        r"$.store.book[last].test_key\uD83D\uDC8E测试",
        r#"$.store.book[0,1, last - 2].price"#,
        r#"$.store.book[0,1 to last-1]"#,
        r#"$."store"."book""#,
        r#"$."st\"ore"."book\uD83D\uDC8E""#,
        r#"$[*].book.price ? (@ == 10)"#,
        r#"$.store.book?(@.price > 10).title"#,
        r#"$.store.book?(@.price < $.expensive).price"#,
        r#"$.store.book?(@.price < 10 && @.category == "fiction")"#,
        r#"$.store.book?(@.price > 10 || @.category == "reference")"#,
        r#"$.store.book?(@.price > 20 && (@.category == "reference" || @.category == "fiction"))"#,
        // compatible with Snowflake style path
        r#"[1][2]"#,
        r#"["k1"]["k2"]"#,
        r#"k1.k2:k3"#,
        r#"k1["k2"][1]"#,
        // predicates
        r#"$ > 1"#,
        r#"$.* == 0"#,
        r#"$[*] > 1"#,
        r#"$.a > $.b"#,
        r#"$.price > 10 || $.category == "reference""#,
        // exists expression
        r#"$.store.book?(exists(@.price?(@ > 20)))"#,
        r#"$.store?(exists(@.book?(exists(@.category?(@ == "fiction")))))"#,
        r#"$.store.book?(@ starts with "Nigel")"#,
        r#"$[*] ? (@.job == null) .name"#,
        // arithmetic functions
        r#"$.phones[0].number + 3"#,
        r#"7 - $[0]"#,
        r#"- $.phones[0].number"#,
    ];

    for case in cases {
        let json_path = parse_json_path(case.as_bytes()).unwrap();

        writeln!(file, "---------- Input ----------").unwrap();
        writeln!(file, "{case}").unwrap();
        writeln!(file, "---------- Output ---------").unwrap();
        writeln!(file, "{json_path}").unwrap();
        writeln!(file, "---------- AST ------------").unwrap();
        writeln!(file, "{json_path:#?}").unwrap();
        writeln!(file, "\n").unwrap();
    }
}

#[rstest]
#[case::decimal("0.9", Number::Float64(0.9))]
#[case::large_decimal("1234.5", Number::Float64(1234.5))]
#[case::negative_decimal("-0.5", Number::Float64(-0.5))]
#[case::leading_dot(".9", Number::Float64(0.9))]
#[case::negative_leading_dot("-.5", Number::Float64(-0.5))]
#[case::trailing_dot("1.", Number::Float64(1.0))]
#[case::exponent("9E-1", Number::Float64(0.9))]
#[case::lowercase_exponent("9e-1", Number::Float64(0.9))]
#[case::positive_exponent("1e+3", Number::Float64(1000.0))]
#[case::decimal_exponent("1.25e2", Number::Float64(125.0))]
#[case::negative_exponent("-9E-1", Number::Float64(-0.9))]
#[case::zero("0", Number::UInt64(0))]
#[case::positive_integer("42", Number::UInt64(42))]
#[case::negative_integer("-42", Number::Int64(-42))]
#[case::exact_unsigned("9007199254740993", Number::UInt64(9007199254740993))]
#[case::exact_signed("-9007199254740993", Number::Int64(-9007199254740993))]
#[case::max_unsigned("18446744073709551615", Number::UInt64(u64::MAX))]
#[case::min_signed("-9223372036854775808", Number::Int64(i64::MIN))]
#[case::unsigned_overflow("18446744073709551616", Number::Float64(18446744073709551616.0))]
#[case::signed_overflow("-9223372036854775809", Number::Float64(-9223372036854775809.0))]
fn test_json_path_number(#[case] literal: &str, #[case] expected: Number) {
    let path = format!("$[*] ? (@.value == {literal})");
    let json_path = parse_json_path(path.as_bytes()).unwrap();
    let Path::FilterExpr(expr) = &json_path.paths[2] else {
        panic!("expected a filter expression: {json_path:?}");
    };
    let Expr::BinaryOp { right, .. } = expr.as_ref() else {
        panic!("expected a comparison: {expr:?}");
    };
    let Expr::Value(value) = right.as_ref() else {
        panic!("expected a literal: {right:?}");
    };
    let PathValue::Number(actual) = value.as_ref() else {
        panic!("expected a number: {value:?}");
    };
    // Number equality compares across variants; integer literals must keep their type.
    assert_eq!(
        std::mem::discriminant(actual),
        std::mem::discriminant(&expected)
    );
    assert_eq!(actual, &expected);
}

#[test]
fn test_json_path_error() {
    let cases = &[
        r#"$.["#,
        r#"$X"#,
        r#"$."#,
        r#"$.prop."#,
        r#"$.prop+."#,
        r#"$.."#,
        r#"$.prop.."#,
        r#"$.foo bar"#,
        r#"$[0, 1, 2 4]"#,
        r#"$['1','2',]"#,
        r#"$['1', ,'3']"#,
        r#"$['aaa'}'bbb']"#,
        r#"@ > 10"#,
        r#"$[*] ? (@.value > 0.9.1)"#,
        r#"$[*] ? (@.value > 9e)"#,
        r#"$[*] ? (@.value > 9E-)"#,
        r#"$[*] ? (@.value > 1.2e+)"#,
        r#"$[*] ? (@.value > 0.9abc)"#,
    ];

    for case in cases {
        let res = parse_json_path(case.as_bytes());
        assert_eq!(res, Err(Error::InvalidJsonPath), "{case}");
    }
}
