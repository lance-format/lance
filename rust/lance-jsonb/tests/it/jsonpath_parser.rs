// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::io::Write;

use goldenfile::Mint;
use lance_jsonb::jsonpath::parse_json_path;

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
    ];

    for case in cases {
        let res = parse_json_path(case.as_bytes());
        assert!(res.is_err());
    }
}
