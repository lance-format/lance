// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;

use ethnum::I256;
use lance_jsonb::Date;
use lance_jsonb::Decimal64;
use lance_jsonb::Decimal128;
use lance_jsonb::Decimal256;
use lance_jsonb::Interval;
use lance_jsonb::Number;
use lance_jsonb::OwnedJsonb;
use lance_jsonb::RawJsonb;
use lance_jsonb::Timestamp;
use lance_jsonb::TimestampTz;
use lance_jsonb::Value;
use lance_jsonb::jsonpath::Selector;
use lance_jsonb::jsonpath::parse_json_path;
use lance_jsonb::parse_value;

#[test]
fn test_path_exists() {
    let sources = vec![
        (r#"{"a":1,"b":2}"#, r#"$.a"#, true),
        (r#"{"a":1,"b":2}"#, r#"$.c"#, false),
        (r#"{"a":1,"b":2}"#, r#"$.a ? (@ == 1)"#, true),
        (r#"{"a":1,"b":2}"#, r#"$.a ? (@ > 1)"#, false),
        (r#"{"a":1,"b":[1,2,3]}"#, r#"$.b[0]"#, true),
        (r#"{"a":1,"b":[1,2,3]}"#, r#"$.b[3]"#, false),
        (
            r#"{"a":1,"b":[1,2,3]}"#,
            r#"$.b[1 to last] ? (@ >=2 && @ <=3)"#,
            true,
        ),
        // predicates always return true in path_exists.
        (r#"{"a":1,"b":[1,2,3]}"#, r#"$.b[1 to last] > 10"#, true),
        (r#"{"a":1,"b":[1,2,3]}"#, r#"$.b[1 to last] > 1"#, true),
    ];
    for (json, path, expect) in sources {
        let owned_jsonb = json.parse::<OwnedJsonb>().unwrap();
        let raw_jsonb = owned_jsonb.as_raw();
        let json_path = parse_json_path(path.as_bytes()).unwrap();
        let res = Selector::new(raw_jsonb).exists(&json_path);
        assert_eq!(res, Ok(expect));
    }
}

#[test]
fn test_path_exists_expr() {
    let source = r#"{"items": [
        {"id": 0, "name": "Andrew", "car": "Volvo"},
        {"id": 1, "name": "Fred", "car": "BMW"},
        {"id": 2, "name": "James"},
        {"id": 3, "name": "Ken"}
    ]}"#;
    let paths = vec![
        (
            "$.items[*]?(exists($.items))",
            r#"[
                {"id": 0, "name": "Andrew", "car": "Volvo"},
                {"id": 1, "name": "Fred", "car": "BMW"},
                {"id": 2, "name": "James"},
                {"id": 3, "name": "Ken"}
            ]"#,
        ),
        (
            "$.items[*]?(exists(@.car))",
            r#"[
                {"id": 0, "name": "Andrew", "car": "Volvo"},
                {"id": 1, "name": "Fred", "car": "BMW"}
            ]"#,
        ),
        (
            r#"$.items[*]?(exists(@.car?(@ == "Volvo")))"#,
            r#"[
                {"id": 0, "name": "Andrew", "car": "Volvo"}
            ]"#,
        ),
        (
            r#"$.items[*]?(exists(@.car) && @.id >= 1)"#,
            r#"[
                {"id": 1, "name": "Fred", "car": "BMW"}
            ]"#,
        ),
        (
            r#"$ ? (exists(@.items[*]?(exists(@.car))))"#,
            r#"[{"items": [
                {"id": 0, "name": "Andrew", "car": "Volvo"},
                {"id": 1, "name": "Fred", "car": "BMW"},
                {"id": 2, "name": "James"},
                {"id": 3, "name": "Ken"}
            ]}]"#,
        ),
        (
            r#"$ ? (exists(@.items[*]?(exists(@.car) && @.id == 5)))"#,
            r#"[]"#,
        ),
    ];

    let owned_jsonb = source.parse::<OwnedJsonb>().unwrap();
    let raw_jsonb = owned_jsonb.as_raw();
    for (path, expected) in paths {
        let json_path = parse_json_path(path.as_bytes()).unwrap();
        let values = Selector::new(raw_jsonb).select_values(&json_path).unwrap();
        let actual = values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let expected_buf = parse_value(expected.as_bytes()).unwrap().to_vec();
        assert_eq!(
            format!("[{actual}]"),
            RawJsonb::new(&expected_buf).to_string()
        );
    }
}

#[test]
fn test_select_by_path() {
    let source = r#"{"name":"Fred","phones":[{"type":"home","number":3720453},{"type":"work","number":5062051}],"car_no":123,"测试\"\uD83D\uDC8E":"ab","numbers":[2,3,4],"key":null}"#;

    let paths = vec![
        (r#"$.name"#, vec![r#""Fred""#]),
        (
            r#"$.phones"#,
            vec![r#"[{"type":"home","number":3720453},{"type":"work","number":5062051}]"#],
        ),
        (r#"$.phones.*"#, vec![]),
        (
            r#"$.phones[*]"#,
            vec![
                r#"{"type":"home","number":3720453}"#,
                r#"{"type":"work","number":5062051}"#,
            ],
        ),
        (
            r#"$.phones.**"#,
            vec![
                r#"[{"type":"home","number":3720453},{"type":"work","number":5062051}]"#,
                r#"{"type":"home","number":3720453}"#,
                r#"3720453"#,
                r#""home""#,
                r#"{"type":"work","number":5062051}"#,
                r#"5062051"#,
                r#""work""#,
            ],
        ),
        (
            r#"$.phones.**{1 to last}"#,
            vec![
                r#"{"type":"home","number":3720453}"#,
                r#"3720453"#,
                r#""home""#,
                r#"{"type":"work","number":5062051}"#,
                r#"5062051"#,
                r#""work""#,
            ],
        ),
        (r#"$.phones[0].*"#, vec![r#"3720453"#, r#""home""#]),
        (r#"$.phones[0].type"#, vec![r#""home""#]),
        (r#"$.phones[*].type[*]"#, vec![r#""home""#, r#""work""#]),
        (
            r#"$.phones[0 to last].number"#,
            vec![r#"3720453"#, r#"5062051"#],
        ),
        (
            r#"$.phones[0 to last]?(4 == 4)"#,
            vec![
                r#"{"type":"home","number":3720453}"#,
                r#"{"type":"work","number":5062051}"#,
            ],
        ),
        (
            r#"$.phones[0 to last]?(@.type == "home")"#,
            vec![r#"{"type":"home","number":3720453}"#],
        ),
        (
            r#"$.phones[0 to last]?(@.number == 3720453)"#,
            vec![r#"{"type":"home","number":3720453}"#],
        ),
        (
            r#"$.phones[0 to last]?(@.number == 3720453 || @.type == "work")"#,
            vec![
                r#"{"type":"home","number":3720453}"#,
                r#"{"type":"work","number":5062051}"#,
            ],
        ),
        (
            r#"$.phones[0 to last]?(@.number == 3720453 && @.type == "work")"#,
            vec![],
        ),
        (
            r#"$.car_no?($.name == "Fred" && $.car_no != null)"#,
            vec![r#"123"#],
        ),
        (r#"$.car_no"#, vec![r#"123"#]),
        (r#"$.测试\"\uD83D\uDC8E"#, vec![r#""ab""#]),
        // predicates return the result of the filter expression.
        (r#"$.phones[0 to last].number == 3720453"#, vec!["true"]),
        (r#"$.phones[0 to last].type == "workk""#, vec!["false"]),
        (r#"$.name == "Fred" && $.car_no == 123"#, vec!["true"]),
        (
            r#"$.phones[*] ? (@.type starts with "ho")"#,
            vec![r#"{"type":"home","number":3720453}"#],
        ),
        // arithmetic functions
        (r#"$.phones[0].number + 3"#, vec![r#"3720456"#]),
        (r#"$.phones[0].number % 10"#, vec![r#"3"#]),
        (r#"7 - $.phones[1].number"#, vec![r#"-5062044"#]),
        (r#"+$.numbers"#, vec![r#"2"#, r#"3"#, r#"4"#]),
        (r#"-$.numbers"#, vec![r#"-2"#, r#"-3"#, r#"-4"#]),
        (r#"$.numbers[1] / 2"#, vec![r#"1.5"#]),
    ];

    let owned_jsonb = source.parse::<OwnedJsonb>().unwrap();
    let raw_jsonb = owned_jsonb.as_raw();
    for (path, expects) in paths {
        let json_path = parse_json_path(path.as_bytes()).unwrap();
        let res = Selector::new(raw_jsonb).select_values(&json_path);
        assert!(res.is_ok());
        let owned_jsonbs = res.unwrap();
        assert_eq!(owned_jsonbs.len(), expects.len());
        for (owned_jsonb, expect) in owned_jsonbs.into_iter().zip(expects.iter()) {
            let expected_buf = parse_value(expect.as_bytes()).unwrap().to_vec();
            assert_eq!(owned_jsonb.as_raw(), RawJsonb::new(&expected_buf));
        }
    }
}

#[test]
fn test_get_by_index() {
    let sources = vec![
        (r#"1234"#, 0, None),
        (r#"[]"#, 0, None),
        (r#"[1,2,3]"#, 1, Some(Value::Number(Number::UInt64(2)))),
        (r#"["a","b","c"]"#, 0, Some(Value::String(Cow::from("a")))),
    ];

    for (s, idx, expect) in sources {
        let owned_jsonb = s.parse::<OwnedJsonb>().unwrap();
        let raw_jsonb = owned_jsonb.as_raw();
        let res = raw_jsonb.get_by_index(idx);
        assert!(res.is_ok());
        let owned_jsonb_opt = res.unwrap();
        match expect {
            Some(expect) => {
                assert!(owned_jsonb_opt.is_some());
                let owned_jsonb = owned_jsonb_opt.unwrap();
                let expected_buf = expect.to_vec();
                assert_eq!(owned_jsonb.to_vec(), expected_buf);
            }
            None => assert_eq!(owned_jsonb_opt, None),
        }
    }
}

#[test]
fn test_get_by_name() {
    let sources = vec![
        (r#"true"#, "a".to_string(), None),
        (r#"[1,2,3]"#, "a".to_string(), None),
        (r#"{"a":"v1","b":[1,2,3]}"#, "k".to_string(), None),
        (
            r#"{"Aa":"v1", "aA":"v2", "aa":"v3"}"#,
            "aa".to_string(),
            Some(Value::String(Cow::from("v3"))),
        ),
        (
            r#"{"Aa":"v1", "aA":"v2", "aa":"v3"}"#,
            "AA".to_string(),
            None,
        ),
    ];

    for (s, name, expect) in sources {
        let owned_jsonb = s.parse::<OwnedJsonb>().unwrap();
        let raw_jsonb = owned_jsonb.as_raw();
        let res = raw_jsonb.get_by_name(&name, false);
        assert!(res.is_ok());
        let owned_jsonb_opt = res.unwrap();
        match expect {
            Some(expect) => {
                assert!(owned_jsonb_opt.is_some());
                let owned_jsonb = owned_jsonb_opt.unwrap();
                let expected_buf = expect.to_vec();
                assert_eq!(owned_jsonb.to_vec(), expected_buf);
            }
            None => assert_eq!(owned_jsonb_opt, None),
        }
    }
}

#[test]
fn test_get_by_name_ignore_case() {
    let sources = vec![
        (r#"true"#, "a".to_string(), None),
        (r#"[1,2,3]"#, "a".to_string(), None),
        (r#"{"a":"v1","b":[1,2,3]}"#, "k".to_string(), None),
        (
            r#"{"Aa":"v1", "aA":"v2", "aa":"v3"}"#,
            "aa".to_string(),
            Some(Value::String(Cow::from("v3"))),
        ),
        (
            r#"{"Aa":"v1", "aA":"v2", "aa":"v3"}"#,
            "AA".to_string(),
            Some(Value::String(Cow::from("v1"))),
        ),
    ];

    for (s, name, expect) in sources {
        let owned_jsonb = s.parse::<OwnedJsonb>().unwrap();
        let raw_jsonb = owned_jsonb.as_raw();
        let res = raw_jsonb.get_by_name(&name, true);
        assert!(res.is_ok());
        let owned_jsonb_opt = res.unwrap();
        match expect {
            Some(expect) => {
                assert!(owned_jsonb_opt.is_some());
                let owned_jsonb = owned_jsonb_opt.unwrap();
                let expected_buf = expect.to_vec();
                assert_eq!(owned_jsonb.to_vec(), expected_buf);
            }
            None => assert_eq!(owned_jsonb_opt, None),
        }
    }
}

#[test]
fn test_compare() {
    let sources = vec![
        (r#"null"#, r#"null"#, Ordering::Equal),
        (r#"null"#, r#"[1,2,3]"#, Ordering::Greater),
        (r#"null"#, r#"{"k":"v"}"#, Ordering::Greater),
        (r#"null"#, r#"123.45"#, Ordering::Greater),
        (r#"null"#, r#""abcd""#, Ordering::Greater),
        (r#"null"#, r#"true"#, Ordering::Greater),
        (r#"null"#, r#"false"#, Ordering::Greater),
        (r#""abcd""#, r#"null"#, Ordering::Less),
        (r#""abcd""#, r#""def""#, Ordering::Less),
        (r#""abcd""#, r#"123.45"#, Ordering::Greater),
        (r#""abcd""#, r#"true"#, Ordering::Greater),
        (r#""abcd""#, r#"false"#, Ordering::Greater),
        (r#"123"#, r#"12.3"#, Ordering::Greater),
        (r#"123"#, r#"123"#, Ordering::Equal),
        (r#"123"#, r#"456.7"#, Ordering::Less),
        (r#"12.3"#, r#"12"#, Ordering::Greater),
        (r#"-12.3"#, r#"12"#, Ordering::Less),
        (r#"123"#, r#"true"#, Ordering::Greater),
        (r#"123"#, r#"false"#, Ordering::Greater),
        (r#"true"#, r#"true"#, Ordering::Equal),
        (r#"true"#, r#"false"#, Ordering::Greater),
        (r#"false"#, r#"true"#, Ordering::Less),
        (r#"false"#, r#"false"#, Ordering::Equal),
        (r#"[1,2,3]"#, r#"null"#, Ordering::Less),
        (r#"[1,2,3]"#, r#"[1,2,3]"#, Ordering::Equal),
        (r#"[1,2,3]"#, r#"[1,2]"#, Ordering::Greater),
        (r#"[1,2,3]"#, r#"[]"#, Ordering::Greater),
        (r#"[1,2,3]"#, r#"[3]"#, Ordering::Less),
        (r#"[1,2,3]"#, r#"["a"]"#, Ordering::Less),
        (r#"[1,2,3]"#, r#"[true]"#, Ordering::Greater),
        (r#"[1,2,3]"#, r#"[1,2,3,4]"#, Ordering::Less),
        (r#"[1,2,3]"#, r#"{"k":"v"}"#, Ordering::Greater),
        (r#"[1,2,3]"#, r#""abcd""#, Ordering::Greater),
        (r#"[1,2,3]"#, r#"1234"#, Ordering::Greater),
        (r#"[1,2,3]"#, r#"12.34"#, Ordering::Greater),
        (r#"[1,2,3]"#, r#"true"#, Ordering::Greater),
        (r#"[1,2,3]"#, r#"false"#, Ordering::Greater),
        (r#"{"k1":"v1","k2":"v2"}"#, r#"null"#, Ordering::Less),
        (r#"{"k1":"v1","k2":"v2"}"#, r#"[1,2,3]"#, Ordering::Less),
        (
            r#"{"k1":"v1","k2":"v2"}"#,
            r#"{"k1":"v1","k2":"v2"}"#,
            Ordering::Equal,
        ),
        (
            r#"{"k1":"v1","k2":"v2"}"#,
            r#"{"k":"v1","k2":"v2"}"#,
            Ordering::Greater,
        ),
        (
            r#"{"k1":"v1","k2":"v2"}"#,
            r#"{"k1":"a1","k2":"v2"}"#,
            Ordering::Greater,
        ),
        (r#"{"k1":123}"#, r#"{"k1":123,"k2":111}"#, Ordering::Less),
        (r#"{"k1":123,"k2":111}"#, r#"{"k1":123}"#, Ordering::Greater),
        (r#"{"k1":"v1","k2":"v2"}"#, r#"{"a":1}"#, Ordering::Greater),
        (r#"{"k1":"v1","k2":"v2"}"#, r#"{}"#, Ordering::Greater),
        (r#"{"k1":"v1","k2":"v2"}"#, r#""ab""#, Ordering::Greater),
        (r#"{"k1":"v1","k2":"v2"}"#, r#"123"#, Ordering::Greater),
        (r#"{"k1":"v1","k2":"v2"}"#, r#"12.34"#, Ordering::Greater),
        (r#"{"k1":"v1","k2":"v2"}"#, r#"true"#, Ordering::Greater),
        (r#"{"k1":"v1","k2":"v2"}"#, r#"false"#, Ordering::Greater),
    ];

    for (left, right, expected) in sources {
        let left_owned_jsonb = left.parse::<OwnedJsonb>().unwrap();
        let left_raw_jsonb = left_owned_jsonb.as_raw();
        let right_owned_jsonb = right.parse::<OwnedJsonb>().unwrap();
        let right_raw_jsonb = right_owned_jsonb.as_raw();

        let res = left_raw_jsonb.cmp(&right_raw_jsonb);
        assert_eq!(res, expected);
    }
}

#[test]
fn test_to_type() {
    let sources = vec![
        (r#"null"#, None, None, None, None),
        (
            r#"true"#,
            Some(true),
            Some(1_i64),
            Some(1_f64),
            Some("true".to_string()),
        ),
        (
            r#"false"#,
            Some(false),
            Some(0_i64),
            Some(0_f64),
            Some("false".to_string()),
        ),
        (
            r#"1"#,
            None,
            Some(1_i64),
            Some(1_f64),
            Some("1".to_string()),
        ),
        (
            r#"-2"#,
            None,
            Some(-2_i64),
            Some(-2_f64),
            Some("-2".to_string()),
        ),
        (
            r#"1.2"#,
            None,
            Some(1_i64),
            Some(1.2_f64),
            Some("1.2".to_string()),
        ),
        (
            r#"1.5"#,
            None,
            Some(2_i64),
            Some(1.5_f64),
            Some("1.5".to_string()),
        ),
        (
            r#"-1.5"#,
            None,
            Some(-2_i64),
            Some(-1.5_f64),
            Some("-1.5".to_string()),
        ),
        (
            r#"1e-1"#,
            None,
            Some(0_i64),
            Some(0.1_f64),
            Some("0.1".to_string()),
        ),
        (
            r#"1.5e0"#,
            None,
            Some(2_i64),
            Some(1.5_f64),
            Some("1.5".to_string()),
        ),
        (
            r#""true""#,
            Some(true),
            None,
            None,
            Some("true".to_string()),
        ),
        (
            r#""false""#,
            Some(false),
            None,
            None,
            Some("false".to_string()),
        ),
        (r#""abcd""#, None, None, None, Some("abcd".to_string())),
        (
            r#""123.1""#,
            None,
            Some(123_i64),
            Some(123.1_f64),
            Some("123.1".to_string()),
        ),
        (
            r#""123.5""#,
            None,
            Some(124_i64),
            Some(123.5_f64),
            Some("123.5".to_string()),
        ),
        (
            r#""-1.5""#,
            None,
            Some(-2_i64),
            Some(-1.5_f64),
            Some("-1.5".to_string()),
        ),
        (
            r#""1.5e0""#,
            None,
            Some(2_i64),
            Some(1.5_f64),
            Some("1.5e0".to_string()),
        ),
        (
            r#""-1.5e0""#,
            None,
            Some(-2_i64),
            Some(-1.5_f64),
            Some("-1.5e0".to_string()),
        ),
    ];

    for (s, expect_bool, expect_i64, expect_f64, expect_str) in sources {
        let owned_jsonb = s.parse::<OwnedJsonb>().unwrap();
        let raw_jsonb = owned_jsonb.as_raw();

        let res = raw_jsonb.to_bool();
        match expect_bool {
            Some(expect) => {
                assert!(res.is_ok());
                assert_eq!(res.unwrap(), expect);
            }
            None => assert!(res.is_err()),
        }
        let res = raw_jsonb.to_i64();
        match expect_i64 {
            Some(expect) => {
                assert!(res.is_ok());
                assert_eq!(res.unwrap(), expect);
            }
            None => assert!(res.is_err()),
        }
        let res = raw_jsonb.to_f64();
        match expect_f64 {
            Some(expect) => {
                assert!(res.is_ok());
                assert_eq!(res.unwrap(), expect);
            }
            None => assert!(res.is_err()),
        }
        let res = raw_jsonb.to_str();
        match expect_str {
            Some(expect) => {
                assert!(res.is_ok());
                assert_eq!(res.unwrap(), expect);
            }
            None => assert!(res.is_err()),
        }
    }
}

#[test]
fn test_to_string() {
    let sources = vec![
        (r#"null"#, r#"null"#),
        (r#"true"#, r#"true"#),
        (r#"false"#, r#"false"#),
        (r#"1234567"#, r#"1234567"#),
        (r#"-1234567"#, r#"-1234567"#),
        (r#"123.4567"#, r#"123.4567"#),
        (r#""abcdef""#, r#""abcdef""#),
        (r#""ab\n\"\uD83D\uDC8E测试""#, r#""ab\n\"💎测试""#),
        (r#""မြန်မာဘာသာ""#, r#""မြန်မာဘာသာ""#),
        (r#""⚠️✅❌""#, r#""⚠️✅❌""#),
        (r#"[1,2,3,4]"#, r#"[1,2,3,4]"#),
        (
            r#"["a","b",true,false,[1,2,3],{"a":"b"}]"#,
            r#"["a","b",true,false,[1,2,3],{"a":"b"}]"#,
        ),
        (
            r#"{"k1":"v1","k2":[1,2,3],"k3":{"a":"b"}}"#,
            r#"{"k1":"v1","k2":[1,2,3],"k3":{"a":"b"}}"#,
        ),
    ];
    for (s, expect) in sources {
        let owned_jsonb = s.parse::<OwnedJsonb>().unwrap();
        let raw_jsonb = owned_jsonb.as_raw();
        let res = raw_jsonb.to_string();
        assert_eq!(res, expect);
    }

    let extension_sources = vec![
        (Value::Binary(&[97, 98, 99]), r#""616263""#),
        (Value::Binary(&[]), r#""""#),
        (Value::Date(Date { value: 90570 }), r#""2217-12-22""#),
        (Value::Date(Date { value: -1 }), r#""1969-12-31""#),
        (
            Value::Timestamp(Timestamp {
                value: 190390000000,
            }),
            r#""1970-01-03 04:53:10.000000""#,
        ),
        (
            Value::Timestamp(Timestamp { value: 0 }),
            r#""1970-01-01 00:00:00.000000""#,
        ),
        (
            Value::TimestampTz(TimestampTz {
                offset: 8 * 3600,
                value: 190390000000,
            }),
            r#""1970-01-03 12:53:10.000000 +0800""#,
        ),
        (
            Value::TimestampTz(TimestampTz {
                offset: 90 * 60,
                value: 0,
            }),
            r#""1970-01-01 01:30:00.000000 +0130""#,
        ),
        (
            Value::Interval(Interval {
                months: 10,
                days: 20,
                micros: 300000000,
            }),
            r#""10 months 20 days 00:05:00""#,
        ),
        (
            Value::Interval(Interval {
                months: -14,
                days: -3,
                micros: -90000000,
            }),
            r#""-1 year -2 months -3 days -00:01:30""#,
        ),
        (
            Value::Interval(Interval {
                months: -25,
                days: -1,
                micros: 0,
            }),
            r#""-2 years -1 month -1 day""#,
        ),
        (
            Value::Interval(Interval {
                months: 0,
                days: -2,
                micros: 5_000_000,
            }),
            r#""-2 days 00:00:05""#,
        ),
        (
            Value::Interval(Interval {
                months: 0,
                days: 0,
                micros: 0,
            }),
            r#""00:00:00""#,
        ),
        (
            Value::Number(Number::Decimal128(Decimal128 {
                scale: 2,
                value: 1234,
            })),
            r#"12.34"#,
        ),
        (
            Value::Number(Number::Decimal64(Decimal64 {
                scale: 2,
                value: -765,
            })),
            r#"-7.65"#,
        ),
        (
            Value::Number(Number::Decimal256(Decimal256 {
                scale: 2,
                value: I256::new(981724),
            })),
            r#"9817.24"#,
        ),
        (
            Value::Array(vec![
                Value::Binary(&[97, 98, 99]),
                Value::Date(Date { value: 90570 }),
                Value::Timestamp(Timestamp {
                    value: 190390000000,
                }),
                Value::TimestampTz(TimestampTz {
                    offset: 8 * 3600,
                    value: 190390000000,
                }),
                Value::Interval(Interval {
                    months: 10,
                    days: 20,
                    micros: 300000000,
                }),
                Value::Number(Number::Decimal128(Decimal128 {
                    scale: 2,
                    value: 1234,
                })),
                Value::Number(Number::Decimal256(Decimal256 {
                    scale: 2,
                    value: I256::new(981724),
                })),
            ]),
            r#"["616263","2217-12-22","1970-01-03 04:53:10.000000","1970-01-03 12:53:10.000000 +0800","10 months 20 days 00:05:00",12.34,9817.24]"#,
        ),
        (
            Value::Object(BTreeMap::from([
                ("k1".to_string(), Value::Binary(&[97, 98, 99])),
                ("k2".to_string(), Value::Date(Date { value: 90570 })),
                (
                    "k3".to_string(),
                    Value::Timestamp(Timestamp {
                        value: 190390000000,
                    }),
                ),
                (
                    "k4".to_string(),
                    Value::TimestampTz(TimestampTz {
                        offset: 8 * 3600,
                        value: 190390000000,
                    }),
                ),
                (
                    "k5".to_string(),
                    Value::Interval(Interval {
                        months: 10,
                        days: 20,
                        micros: 300000000,
                    }),
                ),
                (
                    "k6".to_string(),
                    Value::Number(Number::Decimal128(Decimal128 {
                        scale: 2,
                        value: 1234,
                    })),
                ),
                (
                    "k7".to_string(),
                    Value::Number(Number::Decimal256(Decimal256 {
                        scale: 2,
                        value: I256::new(981724),
                    })),
                ),
            ])),
            r#"{"k1":"616263","k2":"2217-12-22","k3":"1970-01-03 04:53:10.000000","k4":"1970-01-03 12:53:10.000000 +0800","k5":"10 months 20 days 00:05:00","k6":12.34,"k7":9817.24}"#,
        ),
    ];

    for (v, expect) in extension_sources {
        let buf = v.to_vec();
        let raw_jsonb = RawJsonb::new(&buf);

        let res = raw_jsonb.to_string();
        assert_eq!(res, expect);
    }
}
