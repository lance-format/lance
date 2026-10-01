// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use crate::RawJsonb;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt::Display;
use std::fmt::Formatter;

use crate::number::Number;

/// Represents a set of JSON Path chains.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonPath<'a> {
    pub paths: Vec<Path<'a>>,
}

impl JsonPath<'_> {}

/// Represents a valid JSON Path.
#[derive(Debug, Clone, PartialEq)]
pub enum Path<'a> {
    /// `$` represents the root node or element.
    Root,
    /// `@` represents the current node or element being processed in the filter expression.
    Current,
    /// `.*` represents selecting all elements in an Object.
    DotWildcard,
    /// `.**` represents recursive selecting all elements in Array and Object.
    /// The optional `RecursiveLevel` indicates the selected levels.
    RecursiveDotWildcard(Option<RecursiveLevel>),
    /// `[*]` represents selecting all elements in an Array.
    BracketWildcard,
    /// `.<name>` represents selecting element that matched the name in an Object, like `$.event`.
    /// The name can also be written as a string literal, allowing the name to contain special characters, like `$." $price"`.
    DotField(Cow<'a, str>),
    /// `:<name>` represents selecting element that matched the name in an Object, like `$:event`.
    ColonField(Cow<'a, str>),
    /// `["<name>"]` represents selecting element that matched the name in an Object, like `$["event"]`.
    ObjectField(Cow<'a, str>),
    /// `[<index1>,<index2>,..]` represents selecting elements specified by the indices in an Array.
    /// There are several forms of index.
    /// 1. A single number representing the 0-based `n-th` element in the Array.
    ///    e.g. `$[0]` represents the first element in an Array.
    /// 2. The keyword `last` represents the last element in the Array,
    ///    and last minus a number represents the n-th element before the last element,
    ///    e.g. `$[last-1]` represents the penultimate element.
    /// 3. The keyword `to` between two numbers represent all elements of a range in an Array,
    ///    e.g. `$[1 to last]` represents all the elements in the Array from the second to the last.
    ///
    /// There can be more than one index, e.g. `$[0, last-1 to last, 5]` represents the first,
    /// the last two, and the sixth element in an Array.
    ArrayIndices(Vec<ArrayIndex>),
    /// `?(<expression>)` represents selecting all elements in an object or array that match the filter expression, like `$.book[?(@.price < 10)]`.
    FilterExpr(Box<Expr<'a>>),
    /// `<expression>` standalone filter expression, like `$.book[*].price > 10`,
    /// and arithmetic expression, like `-$.a[*]` or `$.a + 3`
    Expr(Box<Expr<'a>>),
}

/// Represents the single index in an Array.
#[derive(Debug, Clone, PartialEq)]
pub enum Index {
    /// The 0-based index in an Array.
    Index(i32),
    /// The last n-th index in an Array.
    LastIndex(i32),
}

/// Represents the index in an Array.
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayIndex {
    /// The single number index.
    Index(Index),
    /// The range index between two number.
    Slice((Index, Index)),
}

impl ArrayIndex {
    /// Converts an `ArrayIndex` to a `HashSet` of indices that should be selected from an Array.
    ///
    /// # Arguments
    ///
    /// * `length` - The length of the array.
    ///
    /// # Returns
    ///
    /// A `HashSet<usize>` containing the indices to select.
    pub(crate) fn to_indices(&self, length: usize) -> HashSet<usize> {
        let length = length as i32;

        let mut indices = HashSet::new();
        match self {
            Self::Index(idx) => {
                let idx = Self::convert_index(idx, length);
                if idx >= 0 && idx < length {
                    indices.insert(idx as usize);
                }
            }
            Self::Slice((start, end)) => {
                let start_idx = Self::convert_index(start, length);
                let end_idx = Self::convert_index(end, length);

                let start_idx = if start_idx < 0 { 0 } else { start_idx as usize };
                let end_idx = if end_idx >= length {
                    (length - 1) as usize
                } else {
                    end_idx as usize
                };
                for idx in start_idx..=end_idx {
                    indices.insert(idx);
                }
            }
        }
        indices
    }

    fn convert_index(index: &Index, length: i32) -> i32 {
        match index {
            Index::Index(idx) => *idx,
            Index::LastIndex(idx) => length + *idx - 1,
        }
    }
}

/// Represents the end level in hierarchical structure.
#[derive(Debug, Clone, PartialEq)]
pub enum RecursiveLevelEnd {
    /// Specifies the end of the recursive level.
    Index(u8),
    /// Specifies that the recursion should continue to the last level.
    Last,
}

/// Represents the selected levels in hierarchical structure.
#[derive(Debug, Clone, PartialEq)]
pub struct RecursiveLevel {
    /// The starting level of the recursive level.
    pub start: u8,
    /// The optional end of the recursive level. If None, the level applies only to the start level.
    pub end: Option<RecursiveLevelEnd>,
}

impl RecursiveLevel {
    /// Checks if the current level matches the recursive level.
    ///
    /// # Arguments
    ///
    /// * `level` - The current level in hierarchical structure.
    ///
    /// # Returns
    ///
    /// A tuple (is_match, should_continue):
    /// - is_match: Indicates whether the current level matches the level criteria.
    /// - should_continue: Indicates whether to continue processing data at the next level.
    pub(crate) fn check_recursive_level(&self, level: u8) -> (bool, bool) {
        if let Some(end) = &self.end {
            match end {
                RecursiveLevelEnd::Index(end) => {
                    if level < self.start && self.start <= *end {
                        (false, true)
                    } else if level >= self.start && level <= *end {
                        (true, true)
                    } else {
                        (false, false)
                    }
                }
                RecursiveLevelEnd::Last => {
                    if level < self.start {
                        (false, true)
                    } else {
                        (true, true)
                    }
                }
            }
        } else if level < self.start {
            (false, true)
        } else if level == self.start {
            (true, false)
        } else {
            (false, false)
        }
    }
}

/// Represents a literal value used in filter expression.
#[derive(Debug, Clone)]
pub enum PathValue<'a> {
    /// Null value.
    Null,
    /// Boolean value.
    Boolean(bool),
    /// Number value.
    Number(Number),
    /// UTF-8 string.
    String(Cow<'a, str>),
    /// RawJsonb (Array or Object) value, can't be used for calculation.
    Raw(RawJsonb<'a>),
}

impl PartialOrd for PathValue<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (PathValue::Null, PathValue::Null) => Some(Ordering::Equal),
            (PathValue::Boolean(l), PathValue::Boolean(r)) => l.partial_cmp(r),
            (PathValue::Number(l), PathValue::Number(r)) => l.partial_cmp(r),
            (PathValue::String(l), PathValue::String(r)) => l.partial_cmp(r),
            (_, _) => None,
        }
    }
}

impl PartialEq for PathValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

/// Represents the operators used in filter expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryOperator {
    /// `&&` represents logical And operation.
    And,
    /// `||` represents logical Or operation.
    Or,
    /// `==` represents left is equal to right.
    Eq,
    /// `!=` and `<>` represents left is not equal to right.
    NotEq,
    /// `<` represents left is less than right.
    Lt,
    /// `<=` represents left is less or equal to right.
    Lte,
    /// `>` represents left is greater than right.
    Gt,
    /// `>=` represents left is greater than or equal to right.
    Gte,
    /// `starts with` represents right is an initial substring of left.
    StartsWith,
    /// `Add` represents binary arithmetic + operation.
    Add,
    /// `Subtract` represents binary arithmetic - operation.
    Subtract,
    /// `Multiply` represents binary arithmetic * operation.
    Multiply,
    /// `Divide` represents binary arithmetic / operation.
    Divide,
    /// `Modulo` represents binary arithmetic % operation.
    Modulo,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    /// `Add` represents unary arithmetic + operation.
    Add,
    /// `Subtract` represents unary arithmetic - operation.
    Subtract,
}

/// Represents a filter expression used to filter Array or Object.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr<'a> {
    /// JSON Path chains.
    Paths(Vec<Path<'a>>),
    /// Literal value.
    Value(Box<PathValue<'a>>),
    /// Filter expression that performs a binary operation, returns a boolean value,
    /// Or arithmetic expression that performs an arithmetic operation, returns a number value.
    BinaryOp {
        op: BinaryOperator,
        left: Box<Self>,
        right: Box<Self>,
    },
    /// Unary expression that performs an arithmetic operation.
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Self>,
    },
    /// Filter function, returns a boolean value.
    ExistsFunc(Vec<Path<'a>>),
}

impl Display for JsonPath<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for path in &self.paths {
            write!(f, "{path}")?;
        }
        Ok(())
    }
}

impl Display for Index {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Index(idx) => {
                write!(f, "{idx}")?;
            }
            Self::LastIndex(idx) => {
                write!(f, "last")?;
                match idx.cmp(&0) {
                    Ordering::Greater => {
                        write!(f, "+{idx}")?;
                    }
                    Ordering::Less => {
                        write!(f, "{idx}")?;
                    }
                    Ordering::Equal => {}
                }
            }
        }
        Ok(())
    }
}

impl Display for ArrayIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Index(idx) => {
                write!(f, "{idx}")?;
            }
            Self::Slice((start, end)) => {
                write!(f, "{start} to {end}")?;
            }
        }
        Ok(())
    }
}

impl Display for RecursiveLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if let Some(end_index) = &self.end {
            match end_index {
                RecursiveLevelEnd::Index(end) => {
                    write!(f, "{} to {}", self.start, end)?;
                }
                RecursiveLevelEnd::Last => {
                    write!(f, "{} to last", self.start)?;
                }
            }
        } else {
            write!(f, "{}", self.start)?;
        }
        Ok(())
    }
}

impl Display for Path<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Path::Root => {
                write!(f, "$")?;
            }
            Path::Current => {
                write!(f, "@")?;
            }
            Path::DotWildcard => {
                write!(f, ".*")?;
            }
            Path::RecursiveDotWildcard(level_opt) => {
                write!(f, ".**")?;
                if let Some(level) = level_opt {
                    write!(f, "{{")?;
                    write!(f, "{level}")?;
                    write!(f, "}}")?;
                }
            }
            Path::BracketWildcard => {
                write!(f, "[*]")?;
            }
            Path::ColonField(field) => {
                write!(f, ":{field}")?;
            }
            Path::DotField(field) => {
                write!(f, ".{field}")?;
            }
            Path::ObjectField(field) => {
                write!(f, "[\"{field}\"]")?;
            }
            Path::ArrayIndices(indices) => {
                write!(f, "[")?;
                for (i, index) in indices.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{index}")?;
                }
                write!(f, "]")?;
            }
            Path::FilterExpr(expr) => {
                write!(f, "?({expr})")?;
            }
            Path::Expr(expr) => {
                write!(f, "{expr}")?;
            }
        }
        Ok(())
    }
}

impl Display for PathValue<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PathValue::Null => {
                write!(f, "null")
            }
            PathValue::Boolean(v) => {
                if *v {
                    write!(f, "true")
                } else {
                    write!(f, "false")
                }
            }
            PathValue::Number(v) => {
                write!(f, "{v}")
            }
            PathValue::String(v) => {
                write!(f, "\"{v}\"")
            }
            PathValue::Raw(v) => {
                write!(f, "{}", v.to_string())
            }
        }
    }
}

impl Display for BinaryOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::And => {
                write!(f, "&&")
            }
            Self::Or => {
                write!(f, "||")
            }
            Self::Eq => {
                write!(f, "==")
            }
            Self::NotEq => {
                write!(f, "!=")
            }
            Self::Lt => {
                write!(f, "<")
            }
            Self::Lte => {
                write!(f, "<=")
            }
            Self::Gt => {
                write!(f, ">")
            }
            Self::Gte => {
                write!(f, ">=")
            }
            Self::StartsWith => {
                write!(f, "starts with")
            }
            Self::Add => {
                write!(f, "+")
            }
            Self::Subtract => {
                write!(f, "-")
            }
            Self::Multiply => {
                write!(f, "*")
            }
            Self::Divide => {
                write!(f, "/")
            }
            Self::Modulo => {
                write!(f, "%")
            }
        }
    }
}

impl Display for UnaryOperator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => {
                write!(f, "+")
            }
            Self::Subtract => {
                write!(f, "-")
            }
        }
    }
}

impl Display for Expr<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Paths(paths) => {
                for path in paths {
                    write!(f, "{path}")?;
                }
            }
            Expr::Value(v) => {
                write!(f, "{v}")?;
            }
            Expr::BinaryOp { op, left, right } => {
                if let Expr::BinaryOp { op: left_op, .. } = &**left {
                    if left_op == &BinaryOperator::And || left_op == &BinaryOperator::Or {
                        write!(f, "({left})")?;
                    } else {
                        write!(f, "{left}")?;
                    }
                } else {
                    write!(f, "{left}")?;
                }
                write!(f, " {op} ")?;
                if let Expr::BinaryOp { op: right_op, .. } = &**right {
                    if right_op == &BinaryOperator::And || right_op == &BinaryOperator::Or {
                        write!(f, "({right})")?;
                    } else {
                        write!(f, "{right}")?;
                    }
                } else {
                    write!(f, "{right}")?;
                }
            }
            Expr::UnaryOp { op, operand } => {
                write!(f, "{}{}", op, operand)?;
            }
            Expr::ExistsFunc(paths) => {
                f.write_str("exists(")?;
                for path in paths {
                    write!(f, "{path}")?;
                }
                f.write_str(")")?;
            }
        }
        Ok(())
    }
}
