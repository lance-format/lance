// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::borrow::Cow;

use nom::IResult;
use nom::Parser;
use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::bytes::complete::tag_no_case;
use nom::character::complete::char;
use nom::character::complete::i32;
use nom::character::complete::i64;
use nom::character::complete::multispace0;
use nom::character::complete::u8;
use nom::character::complete::u64;
use nom::combinator::cond;
use nom::combinator::map;
use nom::combinator::map_res;
use nom::combinator::opt;
use nom::combinator::value;
use nom::error::Error as NomError;
use nom::error::ErrorKind;
use nom::multi::many0;
use nom::multi::separated_list1;
use nom::number::complete::double;
use nom::sequence::delimited;
use nom::sequence::pair;
use nom::sequence::preceded;
use nom::sequence::separated_pair;
use nom::sequence::terminated;

use crate::constants::UNICODE_LEN;
use crate::error::Error;
use crate::jsonpath::path::*;
use crate::number::Number;
use crate::util::parse_string;

/// Parsing the input string to JSON Path.
pub fn parse_json_path(input: &[u8]) -> Result<JsonPath<'_>, Error> {
    match json_path(input) {
        Ok((rest, json_path)) => {
            if !rest.is_empty() {
                return Err(Error::InvalidJsonPath);
            }
            Ok(json_path)
        }
        Err(nom::Err::Error(_) | nom::Err::Failure(_)) => Err(Error::InvalidJsonPath),
        Err(nom::Err::Incomplete(_)) => unreachable!(),
    }
}

fn json_path(input: &[u8]) -> IResult<&[u8], JsonPath<'_>> {
    map(
        delimited(multispace0, expr_or_paths, multispace0),
        |paths| JsonPath { paths },
    )
    .parse(input)
}

fn check_escaped(input: &[u8], i: &mut usize) -> bool {
    if *i + 1 >= input.len() {
        return false;
    }
    if input[*i + 1] == b'u' {
        if *i + 5 >= input.len() {
            return false;
        }
        if input[*i + 2] == b'{' {
            if *i + 7 >= input.len() {
                return false;
            }
            *i += UNICODE_LEN + 4;
        } else {
            *i += UNICODE_LEN + 2;
        }
    } else {
        *i += 2;
    }
    true
}

pub fn raw_string(input: &[u8]) -> IResult<&[u8], Cow<'_, str>> {
    let mut i = 0;
    let mut escapes = 0;
    while i < input.len() {
        let c = input[i];
        match c {
            b'\\' => {
                escapes += 1;
                if !check_escaped(input, &mut i) {
                    return Err(nom::Err::Error(NomError::new(input, ErrorKind::Char)));
                }
            }
            b' ' | b',' | b'.' | b':' | b'{' | b'}' | b'[' | b']' | b'(' | b')' | b'?' | b'@'
            | b'$' | b'|' | b'<' | b'>' | b'!' | b'=' | b'+' | b'-' | b'*' | b'/' | b'%' | b'"'
            | b'\'' => {
                break;
            }
            _ => {
                i += 1;
            }
        }
    }
    if i > 0 {
        if escapes == 0 {
            if let Ok(s) = std::str::from_utf8(&input[..i]) {
                return Ok((&input[i..], Cow::Borrowed(s)));
            }
        } else {
            let data = &input[..i];
            let len = i - escapes;
            let mut idx = 0;
            let s = parse_string(data, len, &mut idx)
                .map_err(|_| nom::Err::Error(NomError::new(input, ErrorKind::Char)))?;
            return Ok((&input[i..], Cow::Owned(s)));
        }
    }
    Err(nom::Err::Error(NomError::new(input, ErrorKind::Char)))
}

pub fn string(input: &[u8]) -> IResult<&[u8], Cow<'_, str>> {
    if input.is_empty() || input[0] != b'"' {
        return Err(nom::Err::Error(NomError::new(input, ErrorKind::Char)));
    }
    let mut i = 1;
    let mut escapes = 0;
    while i < input.len() {
        let c = input[i];
        match c {
            b'\\' => {
                escapes += 1;
                if !check_escaped(input, &mut i) {
                    return Err(nom::Err::Error(NomError::new(input, ErrorKind::Char)));
                }
            }
            b'"' => {
                break;
            }
            _ => {
                i += 1;
            }
        }
    }
    if i > 1 {
        if escapes == 0 {
            if let Ok(s) = std::str::from_utf8(&input[1..i]) {
                return Ok((&input[i + 1..], Cow::Borrowed(s)));
            }
        } else {
            let data = &input[1..i];
            let len = i - 1 - escapes;
            let mut idx = 1;
            let s = parse_string(data, len, &mut idx)
                .map_err(|_| nom::Err::Error(NomError::new(input, ErrorKind::Char)))?;
            return Ok((&input[i + 1..], Cow::Owned(s)));
        }
    }
    Err(nom::Err::Error(NomError::new(input, ErrorKind::Char)))
}

fn bracket_wildcard(input: &[u8]) -> IResult<&[u8], ()> {
    value(
        (),
        delimited(
            char('['),
            delimited(multispace0, char('*'), multispace0),
            char(']'),
        ),
    )
    .parse(input)
}

fn recursive_dot_wildcard(input: &[u8]) -> IResult<&[u8], Option<RecursiveLevel>> {
    preceded(tag(".**"), opt(recursive_level)).parse(input)
}

fn recursive_level(input: &[u8]) -> IResult<&[u8], RecursiveLevel> {
    alt((
        delimited(
            char('{'),
            delimited(multispace0, recursive_level_range, multispace0),
            char('}'),
        ),
        map(
            delimited(
                char('{'),
                delimited(multispace0, u8, multispace0),
                char('}'),
            ),
            |s| RecursiveLevel {
                start: s,
                end: None,
            },
        ),
    ))
    .parse(input)
}

fn recursive_level_range(input: &[u8]) -> IResult<&[u8], RecursiveLevel> {
    alt((
        map(
            separated_pair(
                u8,
                delimited(multispace0, tag_no_case("to"), multispace0),
                u8,
            ),
            |(s, e)| RecursiveLevel {
                start: s,
                end: Some(RecursiveLevelEnd::Index(e)),
            },
        ),
        map(
            separated_pair(
                u8,
                delimited(multispace0, tag_no_case("to"), multispace0),
                tag_no_case("last"),
            ),
            |(s, _)| RecursiveLevel {
                start: s,
                end: Some(RecursiveLevelEnd::Last),
            },
        ),
    ))
    .parse(input)
}

fn colon_field(input: &[u8]) -> IResult<&[u8], Cow<'_, str>> {
    alt((preceded(char(':'), string), preceded(char(':'), raw_string))).parse(input)
}

fn dot_field(input: &[u8]) -> IResult<&[u8], Cow<'_, str>> {
    alt((preceded(char('.'), string), preceded(char('.'), raw_string))).parse(input)
}

fn object_field(input: &[u8]) -> IResult<&[u8], Cow<'_, str>> {
    delimited(
        terminated(char('['), multispace0),
        string,
        preceded(multispace0, char(']')),
    )
    .parse(input)
}

fn index(input: &[u8]) -> IResult<&[u8], Index> {
    alt((
        map(i32, Index::Index),
        map(
            preceded(
                separated_pair(tag_no_case("last"), multispace0, char('-')),
                preceded(multispace0, i32),
            ),
            |v| Index::LastIndex(v.saturating_neg()),
        ),
        map(
            preceded(
                separated_pair(tag_no_case("last"), multispace0, char('+')),
                preceded(multispace0, i32),
            ),
            Index::LastIndex,
        ),
        map(tag_no_case("last"), |_| Index::LastIndex(0)),
    ))
    .parse(input)
}

fn array_index(input: &[u8]) -> IResult<&[u8], ArrayIndex> {
    alt((
        map(
            separated_pair(
                index,
                delimited(multispace0, tag_no_case("to"), multispace0),
                index,
            ),
            |(s, e)| ArrayIndex::Slice((s, e)),
        ),
        map(index, ArrayIndex::Index),
    ))
    .parse(input)
}

fn array_indices(input: &[u8]) -> IResult<&[u8], Vec<ArrayIndex>> {
    delimited(
        char('['),
        separated_list1(char(','), delimited(multispace0, array_index, multispace0)),
        char(']'),
    )
    .parse(input)
}

fn inner_path(input: &[u8]) -> IResult<&[u8], Path<'_>> {
    alt((
        map(recursive_dot_wildcard, Path::RecursiveDotWildcard),
        value(Path::DotWildcard, tag(".*")),
        value(Path::BracketWildcard, bracket_wildcard),
        map(colon_field, Path::ColonField),
        map(dot_field, Path::DotField),
        map(array_indices, Path::ArrayIndices),
        map(object_field, Path::ObjectField),
    ))
    .parse(input)
}

// Compatible with Snowflake query syntax, the first field name does not require the leading period
fn pre_path(input: &[u8]) -> IResult<&[u8], Path<'_>> {
    alt((
        value(Path::Root, char('$')),
        map(delimited(multispace0, raw_string, multispace0), |v| {
            Path::DotField(v)
        }),
    ))
    .parse(input)
}

fn path(input: &[u8]) -> IResult<&[u8], Path<'_>> {
    alt((
        map(delimited(multispace0, inner_path, multispace0), |v| v),
        map(delimited(multispace0, filter_expr, multispace0), |v| {
            Path::FilterExpr(Box::new(v))
        }),
    ))
    .parse(input)
}

fn expr_or_paths(input: &[u8]) -> IResult<&[u8], Vec<Path<'_>>> {
    alt((root_expr, paths)).parse(input)
}

fn root_expr(input: &[u8]) -> IResult<&[u8], Vec<Path<'_>>> {
    map(
        delimited(multispace0, |i| expr_or(i, true), multispace0),
        |v| vec![Path::Expr(Box::new(v))],
    )
    .parse(input)
}

fn paths(input: &[u8]) -> IResult<&[u8], Vec<Path<'_>>> {
    map(
        pair(opt(pre_path), many0(path)),
        |(opt_pre_path, mut paths)| {
            if let Some(pre_path) = opt_pre_path {
                paths.insert(0, pre_path);
            }
            paths
        },
    )
    .parse(input)
}

fn expr_paths(input: &[u8], is_root_expr: bool) -> IResult<&[u8], Vec<Path<'_>>> {
    let parse_current = map_res(
        cond(!is_root_expr, value(Path::Current, char('@'))),
        |res| match res {
            Some(v) => Ok(v),
            None => Err(NomError::new(input, ErrorKind::Char)),
        },
    );
    map(
        pair(
            alt((value(Path::Root, char('$')), parse_current)),
            many0(delimited(multispace0, inner_path, multispace0)),
        ),
        |(pre_path, mut paths)| {
            paths.insert(0, pre_path);
            paths
        },
    )
    .parse(input)
}

fn filter_expr(input: &[u8]) -> IResult<&[u8], Expr<'_>> {
    map(
        delimited(
            delimited(char('?'), multispace0, char('(')),
            delimited(multispace0, |i| expr_or(i, false), multispace0),
            char(')'),
        ),
        |v| v,
    )
    .parse(input)
}

fn op(input: &[u8]) -> IResult<&[u8], BinaryOperator> {
    alt((
        value(BinaryOperator::Eq, tag("==")),
        value(BinaryOperator::NotEq, tag("!=")),
        value(BinaryOperator::NotEq, tag("<>")),
        value(BinaryOperator::Lte, tag("<=")),
        value(BinaryOperator::Lt, char('<')),
        value(BinaryOperator::Gte, tag(">=")),
        value(BinaryOperator::Gt, char('>')),
        value(BinaryOperator::StartsWith, tag("starts with")),
        value(BinaryOperator::Add, char('+')),
        value(BinaryOperator::Subtract, char('-')),
        value(BinaryOperator::Multiply, char('*')),
        value(BinaryOperator::Divide, char('/')),
        value(BinaryOperator::Modulo, char('%')),
    ))
    .parse(input)
}

fn unary_arith_op(input: &[u8]) -> IResult<&[u8], UnaryOperator> {
    alt((
        value(UnaryOperator::Add, char('+')),
        value(UnaryOperator::Subtract, char('-')),
    ))
    .parse(input)
}

fn path_value(input: &[u8]) -> IResult<&[u8], PathValue<'_>> {
    alt((
        value(PathValue::Null, tag("null")),
        value(PathValue::Boolean(true), tag("true")),
        value(PathValue::Boolean(false), tag("false")),
        map(u64, |v| PathValue::Number(Number::UInt64(v))),
        map(i64, |v| PathValue::Number(Number::Int64(v))),
        map(double, |v| PathValue::Number(Number::Float64(v))),
        map(string, PathValue::String),
    ))
    .parse(input)
}

fn inner_expr(input: &[u8], is_root_expr: bool) -> IResult<&[u8], Expr<'_>> {
    alt((
        map(|i| expr_paths(i, is_root_expr), Expr::Paths),
        map(path_value, |v| Expr::Value(Box::new(v))),
    ))
    .parse(input)
}

fn expr_atom(input: &[u8], is_root_expr: bool) -> IResult<&[u8], Expr<'_>> {
    alt((
        map(
            pair(
                unary_arith_op,
                delimited(multispace0, |i| inner_expr(i, is_root_expr), multispace0),
            ),
            |(op, operand)| Expr::UnaryOp {
                op,
                operand: Box::new(operand),
            },
        ),
        map(
            pair(
                pair(
                    delimited(multispace0, |i| inner_expr(i, is_root_expr), multispace0),
                    op,
                ),
                delimited(multispace0, |i| inner_expr(i, is_root_expr), multispace0),
            ),
            |((left, op), right)| Expr::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
        ),
        map(
            delimited(
                terminated(char('('), multispace0),
                |i| expr_or(i, is_root_expr),
                preceded(multispace0, char(')')),
            ),
            |expr| expr,
        ),
        map(exists_func, Expr::ExistsFunc),
    ))
    .parse(input)
}

fn exists_func(input: &[u8]) -> IResult<&[u8], Vec<Path<'_>>> {
    preceded(
        tag("exists"),
        preceded(
            multispace0,
            delimited(
                terminated(char('('), multispace0),
                exists_paths,
                preceded(multispace0, char(')')),
            ),
        ),
    )
    .parse(input)
}

fn exists_paths(input: &[u8]) -> IResult<&[u8], Vec<Path<'_>>> {
    map(
        pair(
            alt((
                value(Path::Root, char('$')),
                value(Path::Current, char('@')),
            )),
            many0(path),
        ),
        |(pre, mut paths)| {
            paths.insert(0, pre);
            paths
        },
    )
    .parse(input)
}

fn expr_and(input: &[u8], is_root_expr: bool) -> IResult<&[u8], Expr<'_>> {
    map(
        separated_list1(delimited(multispace0, tag("&&"), multispace0), |i| {
            expr_atom(i, is_root_expr)
        }),
        |exprs| {
            let mut expr = exprs[0].clone();
            for right in exprs.iter().skip(1) {
                expr = Expr::BinaryOp {
                    op: BinaryOperator::And,
                    left: Box::new(expr),
                    right: Box::new(right.clone()),
                };
            }
            expr
        },
    )
    .parse(input)
}

fn expr_or(input: &[u8], is_root_expr: bool) -> IResult<&[u8], Expr<'_>> {
    map(
        separated_list1(delimited(multispace0, tag("||"), multispace0), |i| {
            expr_and(i, is_root_expr)
        }),
        |exprs| {
            let mut expr = exprs[0].clone();
            for right in exprs.iter().skip(1) {
                expr = Expr::BinaryOp {
                    op: BinaryOperator::Or,
                    left: Box::new(expr),
                    right: Box::new(right.clone()),
                };
            }
            expr
        },
    )
    .parse(input)
}
