// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// SPDX-FileCopyrightText: Copyright 2023 Datafuse Labs
// Adapted from databendlabs/jsonb at commit fba895c5ebe77ce2539e187f9c652f51cbf195c3.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::VecDeque;

use crate::Error;
use crate::OwnedJsonb;
use crate::RawJsonb;
use crate::core::ArrayBuilder;
use crate::core::ArrayIterator;
use crate::core::JsonbItem;
use crate::core::JsonbItemType;
use crate::core::ObjectValueIterator;
use crate::error::Result;
use crate::jsonpath::ArrayIndex;
use crate::jsonpath::BinaryOperator;
use crate::jsonpath::Expr;
use crate::jsonpath::JsonPath;
use crate::jsonpath::Path;
use crate::jsonpath::PathValue;
use crate::jsonpath::RecursiveLevel;
use crate::jsonpath::UnaryOperator;
use crate::number::Number;
use crate::value::Value;

#[derive(Debug)]
enum ExprValue<'a> {
    Values(Vec<PathValue<'a>>),
    Value(Box<PathValue<'a>>),
}

impl ExprValue<'_> {
    fn convert_to_number(self) -> Result<Number> {
        match self {
            ExprValue::Values(mut vals) => {
                if vals.len() != 1 {
                    return Err(Error::InvalidJsonPath);
                }
                let val = vals.pop().unwrap();
                match val {
                    PathValue::Number(num) => Ok(num),
                    _ => Err(Error::InvalidJsonPath),
                }
            }
            ExprValue::Value(val) => match *val {
                PathValue::Number(num) => Ok(num),
                _ => Err(Error::InvalidJsonPath),
            },
        }
    }

    fn convert_to_numbers(self) -> Result<Vec<Number>> {
        match self {
            ExprValue::Values(vals) => {
                let mut nums = Vec::with_capacity(vals.len());
                for val in vals {
                    if let PathValue::Number(num) = val {
                        nums.push(num);
                    } else {
                        return Err(Error::InvalidJsonPath);
                    }
                }
                Ok(nums)
            }
            ExprValue::Value(val) => match *val {
                PathValue::Number(num) => Ok(vec![num]),
                _ => Err(Error::InvalidJsonPath),
            },
        }
    }
}

/// Represents the state of a JSON Path selection process.
///
/// It holds the root JSONB value and the intermediate results (items) found during
/// the execution of a `JsonPath`.
pub struct Selector<'a> {
    /// The root JSONB value against which the path is executed.
    root_jsonb: RawJsonb<'a>,
    /// A queue holding the JSONB items that match the path criteria during execution.
    items: VecDeque<JsonbItem<'a>>,
}

impl<'a> Selector<'a> {
    /// Creates a new `Selector` for the given root `RawJsonb`.
    ///
    /// # Arguments
    ///
    /// * `root_jsonb` - The `RawJsonb` data to select from.
    pub fn new(root_jsonb: RawJsonb<'a>) -> Self {
        Self {
            root_jsonb,
            items: VecDeque::new(),
        }
    }

    /// Executes the `JsonPath` and collects all matching items into a `Vec<OwnedJsonb>`.
    ///
    /// This function returns all matching elements as a `Vec<OwnedJsonb>`.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONPath selector.
    /// * `json_path` - The JSONPath expression.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<OwnedJsonb>)` - A vector containing the selected `OwnedJsonb` values.
    /// * `Err(Error)` - If the JSONB data is invalid or if an error occurs during path evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// use lance_jsonb::jsonpath::parse_json_path;
    /// use lance_jsonb::jsonpath::Selector;
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// let jsonb_value = r#"{"a": {"b": [1, 2, 3]}, "c": 4}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = jsonb_value.as_raw();
    /// let mut selector = Selector::new(raw_jsonb);
    ///
    /// let path = parse_json_path("$.a.b[*]".as_bytes()).unwrap();
    /// let result = selector.select_values(&path).unwrap();
    /// assert_eq!(result.len(), 3);
    /// assert_eq!(result[0].to_string(), "1");
    /// assert_eq!(result[1].to_string(), "2");
    /// assert_eq!(result[2].to_string(), "3");
    /// ```
    ///
    /// # See Also
    ///
    /// * `RawJsonb::select_by_path`.
    pub fn select_values(&mut self, json_path: &'a JsonPath<'a>) -> Result<Vec<OwnedJsonb>> {
        self.execute(json_path)?;
        let mut values = Vec::with_capacity(self.items.len());
        while let Some(item) = self.items.pop_front() {
            let value = OwnedJsonb::from_item(item)?;
            values.push(value);
        }
        Ok(values)
    }

    /// Executes the `JsonPath` and returns a single value or an array of values.
    ///
    /// If exactly one element matches, it is returned directly (wrapped in `Some`).
    /// If multiple elements match, they are returned as a JSON array (wrapped in `Some`).
    /// If no elements match, `None` is returned.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONPath selector.
    /// * `json_path` - The JSONPath expression.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(OwnedJsonb))` - A single `OwnedJsonb` containing the matched values.
    /// * `Ok(None)` - The path does not match any values.
    /// * `Err(Error)` - If the JSONB data is invalid or if an error occurs during path evaluation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::jsonpath::parse_json_path;
    /// use lance_jsonb::jsonpath::Selector;
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// let jsonb_value = r#"{"a": [{"b": 1}, {"b": 2}], "c": 3}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = jsonb_value.as_raw();
    /// let mut selector = Selector::new(raw_jsonb);
    ///
    /// let path = parse_json_path("$.c".as_bytes()).unwrap(); // Matches a single value.
    /// let result = selector.select_value(&path).unwrap();
    /// assert_eq!(result.unwrap().to_string(), "3");
    ///
    /// let path = parse_json_path("$.a[*].b".as_bytes()).unwrap(); // Matches multiple values.
    /// let result = selector.select_value(&path).unwrap();
    /// assert_eq!(result.unwrap().to_string(), "[1,2]");
    ///
    /// let path = parse_json_path("$.x".as_bytes()).unwrap(); // No match.
    /// let result = selector.select_value(&path).unwrap();
    /// assert!(result.is_none());
    /// ```
    ///
    /// # See Also
    ///
    /// * `RawJsonb::select_value_by_path`.
    pub fn select_value(&mut self, json_path: &'a JsonPath<'a>) -> Result<Option<OwnedJsonb>> {
        self.execute(json_path)?;
        if self.items.len() > 1 {
            let mut builder = ArrayBuilder::with_capacity(self.items.len());
            while let Some(item) = self.items.pop_front() {
                builder.push_jsonb_item(item);
            }
            let array = builder.build()?;
            Ok(Some(array))
        } else if let Some(item) = self.items.pop_front() {
            let value = OwnedJsonb::from_item(item)?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    /// Executes the `JsonPath` and checks if any item matches.
    ///
    /// # Arguments
    ///
    /// * `self` - The JSONPath selector.
    /// * `json_path` - The JSONPath expression.
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - If the JSON path exists.
    /// * `Ok(false)` - If the JSON path does not exist.
    /// * `Err(Error)` - If the JSONB data is invalid or if an error occurs during path evaluation.
    ///   This could also indicate issues with the `json_path` itself.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lance_jsonb::jsonpath::parse_json_path;
    /// use lance_jsonb::jsonpath::Selector;
    /// use lance_jsonb::OwnedJsonb;
    ///
    /// let jsonb_value = r#"{"a": {"b": [1, 2, 3]}, "c": 4}"#.parse::<OwnedJsonb>().unwrap();
    /// let raw_jsonb = jsonb_value.as_raw();
    /// let mut selector = Selector::new(raw_jsonb);
    ///
    /// // Valid paths
    /// let path1 = parse_json_path("$.a.b[1]".as_bytes()).unwrap();
    /// assert!(selector.exists(&path1).unwrap());
    ///
    /// let path2 = parse_json_path("$.c".as_bytes()).unwrap();
    /// assert!(selector.exists(&path2).unwrap());
    ///
    /// // Invalid paths
    /// let path3 = parse_json_path("$.a.x".as_bytes()).unwrap(); // "x" does not exist
    /// assert!(!selector.exists(&path3).unwrap());
    /// ```
    ///
    /// # See Also
    ///
    /// * `RawJsonb::path_exists`.
    pub fn exists(&mut self, json_path: &'a JsonPath<'a>) -> Result<bool> {
        self.execute(json_path)?;
        Ok(!self.items.is_empty())
    }

    fn execute(&mut self, json_path: &'a JsonPath<'a>) -> Result<()> {
        // add root jsonb
        let root_item = JsonbItem::Raw(self.root_jsonb);
        self.items.clear();
        self.items.push_front(root_item);

        if json_path.paths.len() == 1
            && let Path::Expr(expr) = &json_path.paths[0]
        {
            let root_item = self.items.pop_front().unwrap();
            self.eval_expr(root_item, expr)?;
            return Ok(());
        }
        self.select_by_paths(&json_path.paths)?;

        Ok(())
    }

    fn select_by_paths(&mut self, paths: &'a [Path<'a>]) -> Result<()> {
        if let Some(Path::Current) = paths.first() {
            return Err(Error::InvalidJsonPath);
        }

        for path in paths.iter() {
            match path {
                &Path::Root | &Path::Current => {
                    continue;
                }
                Path::FilterExpr(expr) | Path::Expr(expr) => {
                    let len = self.items.len();
                    for _ in 0..len {
                        let item = self.items.pop_front().unwrap();
                        let res = self.eval_filter_expr(item.clone(), expr)?.unwrap_or(false);
                        if res {
                            self.items.push_back(item);
                        }
                    }
                }
                _ => {
                    self.select_by_path(path)?;
                }
            }
        }
        Ok(())
    }

    fn select_by_path(&mut self, path: &'a Path<'a>) -> Result<bool> {
        if self.items.is_empty() {
            return Ok(false);
        }

        let len = self.items.len();
        for _ in 0..len {
            let item = self.items.pop_front().unwrap();

            match path {
                Path::DotWildcard => {
                    self.select_object_values(item)?;
                }
                Path::RecursiveDotWildcard(index_opt) => {
                    self.recursive_select_values(item, 0, index_opt)?;
                }
                Path::BracketWildcard => {
                    self.select_array_values(item)?;
                }
                Path::ColonField(name) | Path::DotField(name) | Path::ObjectField(name) => {
                    self.select_object_values_by_name(item, name)?;
                }
                Path::ArrayIndices(array_indices) => {
                    self.select_array_values_by_indices(item, array_indices)?;
                }
                _ => todo!(),
            }
        }
        Ok(true)
    }

    fn select_object_values(&mut self, parent_item: JsonbItem<'a>) -> Result<()> {
        let jsonb_item_type = parent_item.jsonb_item_type()?;
        if !matches!(jsonb_item_type, JsonbItemType::Object(_)) {
            return Ok(());
        };

        match parent_item {
            JsonbItem::Raw(raw) => {
                let object_val_iter_opt = ObjectValueIterator::new(raw)?;
                if let Some(mut object_val_iter) = object_val_iter_opt {
                    for result in &mut object_val_iter {
                        let val_item = result?;
                        self.items.push_back(val_item);
                    }
                }
            }
            JsonbItem::Owned(ref owned) => {
                let object_val_iter_opt = ObjectValueIterator::new(owned.as_raw())?;
                if let Some(mut object_val_iter) = object_val_iter_opt {
                    for result in &mut object_val_iter {
                        let val_item = result?;
                        let owned_item = OwnedJsonb::from_item(val_item)?;
                        self.items.push_back(JsonbItem::Owned(owned_item));
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn recursive_select_values(
        &mut self,
        parent_item: JsonbItem<'a>,
        curr_level: u8,
        recursive_level_opt: &Option<RecursiveLevel>,
    ) -> Result<()> {
        let (is_match, should_continue) = if let Some(recursive_level) = recursive_level_opt {
            recursive_level.check_recursive_level(curr_level)
        } else {
            (true, true)
        };
        if is_match {
            self.items.push_back(parent_item.clone());
        }
        if !should_continue {
            return Ok(());
        }

        match parent_item {
            JsonbItem::Raw(raw) => {
                let object_val_iter_opt = ObjectValueIterator::new(raw)?;
                if let Some(mut object_val_iter) = object_val_iter_opt {
                    for result in &mut object_val_iter {
                        let val_item = result?;
                        self.recursive_select_values(
                            val_item,
                            curr_level + 1,
                            recursive_level_opt,
                        )?;
                    }
                }
                let array_iter_opt = ArrayIterator::new(raw)?;
                if let Some(mut array_iter) = array_iter_opt {
                    for item_result in &mut array_iter {
                        let item = item_result?;
                        self.recursive_select_values(item, curr_level + 1, recursive_level_opt)?;
                    }
                }
            }
            JsonbItem::Owned(ref owned) => {
                let object_val_iter_opt = ObjectValueIterator::new(owned.as_raw())?;
                if let Some(mut object_val_iter) = object_val_iter_opt {
                    for result in &mut object_val_iter {
                        let val_item = result?;
                        let owned_item = OwnedJsonb::from_item(val_item)?;
                        self.recursive_select_values(
                            JsonbItem::Owned(owned_item),
                            curr_level + 1,
                            recursive_level_opt,
                        )?;
                    }
                }
                let array_iter_opt = ArrayIterator::new(owned.as_raw())?;
                if let Some(mut array_iter) = array_iter_opt {
                    for item_result in &mut array_iter {
                        let item = item_result?;
                        let owned_item = OwnedJsonb::from_item(item)?;
                        self.recursive_select_values(
                            JsonbItem::Owned(owned_item),
                            curr_level + 1,
                            recursive_level_opt,
                        )?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn select_object_values_by_name(
        &mut self,
        parent_item: JsonbItem<'a>,
        name: &'a str,
    ) -> Result<()> {
        let jsonb_item_type = parent_item.jsonb_item_type()?;
        if !matches!(jsonb_item_type, JsonbItemType::Object(_)) {
            return Ok(());
        };

        let key_name = Cow::Borrowed(name);
        match parent_item {
            JsonbItem::Raw(raw) => {
                if let Some(val_item) =
                    raw.get_object_value_by_key_name(&key_name, |name, key| key.eq(name))?
                {
                    self.items.push_back(val_item);
                }
            }
            JsonbItem::Owned(ref owned) => {
                let raw = owned.as_raw();
                if let Some(val_item) =
                    raw.get_object_value_by_key_name(&key_name, |name, key| key.eq(name))?
                {
                    let owned_item = OwnedJsonb::from_item(val_item)?;
                    self.items.push_back(JsonbItem::Owned(owned_item));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn select_array_values(&mut self, parent_item: JsonbItem<'a>) -> Result<()> {
        let jsonb_item_type = parent_item.jsonb_item_type()?;
        if !matches!(jsonb_item_type, JsonbItemType::Array(_)) {
            // In lax mode, bracket wildcard allow Scalar and Object value.
            self.items.push_back(parent_item);
            return Ok(());
        };

        match parent_item {
            JsonbItem::Raw(raw) => {
                let array_iter_opt = ArrayIterator::new(raw)?;
                if let Some(mut array_iter) = array_iter_opt {
                    for item_result in &mut array_iter {
                        let item = item_result?;
                        self.items.push_back(item);
                    }
                }
            }
            JsonbItem::Owned(ref owned) => {
                let array_iter_opt = ArrayIterator::new(owned.as_raw())?;
                if let Some(mut array_iter) = array_iter_opt {
                    for item_result in &mut array_iter {
                        let item = item_result?;
                        let owned_item = OwnedJsonb::from_item(item)?;
                        self.items.push_back(JsonbItem::Owned(owned_item));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn select_array_values_by_indices(
        &mut self,
        parent_item: JsonbItem<'a>,
        array_indices: &Vec<ArrayIndex>,
    ) -> Result<()> {
        let jsonb_item_type = parent_item.jsonb_item_type()?;
        let JsonbItemType::Array(arr_len) = jsonb_item_type else {
            return Ok(());
        };
        for array_index in array_indices {
            let indices = array_index.to_indices(arr_len);
            if indices.is_empty() {
                continue;
            }
            match parent_item {
                JsonbItem::Raw(raw) => {
                    let array_iter_opt = ArrayIterator::new(raw)?;
                    if let Some(array_iter) = array_iter_opt {
                        for (i, item_result) in &mut array_iter.enumerate() {
                            let item = item_result?;
                            if indices.contains(&i) {
                                self.items.push_back(item);
                            }
                        }
                    }
                }
                JsonbItem::Owned(ref owned) => {
                    let array_iter_opt = ArrayIterator::new(owned.as_raw())?;
                    if let Some(array_iter) = array_iter_opt {
                        for (i, item_result) in &mut array_iter.enumerate() {
                            let item = item_result?;
                            if indices.contains(&i) {
                                let owned_item = OwnedJsonb::from_item(item)?;
                                self.items.push_back(JsonbItem::Owned(owned_item));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn eval_expr(&mut self, item: JsonbItem<'a>, expr: &'a Expr<'a>) -> Result<()> {
        match expr {
            Expr::UnaryOp { op, operand } => {
                let res_items = self.eval_unary_arithmetic_func(item.clone(), op, operand)?;
                for res_item in res_items {
                    self.items.push_back(res_item);
                }
            }
            Expr::BinaryOp { op, left, right } => match op {
                BinaryOperator::Add
                | BinaryOperator::Subtract
                | BinaryOperator::Multiply
                | BinaryOperator::Divide
                | BinaryOperator::Modulo => {
                    let res_items =
                        self.eval_binary_arithmetic_func(item.clone(), op, left, right)?;
                    for res_item in res_items {
                        self.items.push_back(res_item);
                    }
                }
                _ => {
                    let res = self.eval_filter_expr(item, expr)?;
                    let res_item = if let Some(res) = res {
                        JsonbItem::Boolean(res)
                    } else {
                        JsonbItem::Null
                    };
                    self.items.push_back(res_item);
                }
            },
            Expr::ExistsFunc(_) => {
                let res = self.eval_filter_expr(item, expr)?;
                let res_item = if let Some(res) = res {
                    JsonbItem::Boolean(res)
                } else {
                    JsonbItem::Null
                };
                self.items.push_back(res_item);
            }
            Expr::Value(val) => {
                let res_item = self.eval_value(val)?;
                self.items.push_back(res_item);
            }
            Expr::Paths(_) => {
                return Err(Error::InvalidJsonPath);
            }
        }
        Ok(())
    }

    fn eval_unary_arithmetic_func(
        &mut self,
        item: JsonbItem<'a>,
        op: &UnaryOperator,
        operand: &'a Expr<'a>,
    ) -> Result<Vec<JsonbItem<'a>>> {
        let operand = self.convert_expr_val(item, operand)?;
        let Ok(nums) = operand.convert_to_numbers() else {
            return Err(Error::InvalidJsonPath);
        };
        let mut num_vals = Vec::with_capacity(nums.len());
        match op {
            UnaryOperator::Add => {
                for num in nums {
                    let owned_num = OwnedJsonb::new(Value::Number(num).to_vec());
                    num_vals.push(JsonbItem::Owned(owned_num));
                }
            }
            UnaryOperator::Subtract => {
                for num in nums {
                    let neg_num = num.neg()?;
                    let owned_num = OwnedJsonb::new(Value::Number(neg_num).to_vec());
                    num_vals.push(JsonbItem::Owned(owned_num));
                }
            }
        };
        Ok(num_vals)
    }

    fn eval_binary_arithmetic_func(
        &mut self,
        item: JsonbItem<'a>,
        op: &BinaryOperator,
        left: &'a Expr<'a>,
        right: &'a Expr<'a>,
    ) -> Result<Vec<JsonbItem<'a>>> {
        let lhs = self.convert_expr_val(item.clone(), left)?;
        let rhs = self.convert_expr_val(item.clone(), right)?;
        let Ok(lnum) = lhs.convert_to_number() else {
            return Err(Error::InvalidJsonPath);
        };
        let Ok(rnum) = rhs.convert_to_number() else {
            return Err(Error::InvalidJsonPath);
        };

        let num_val = match op {
            BinaryOperator::Add => lnum.add(rnum)?,
            BinaryOperator::Subtract => lnum.sub(rnum)?,
            BinaryOperator::Multiply => lnum.mul(rnum)?,
            BinaryOperator::Divide => lnum.div(rnum)?,
            BinaryOperator::Modulo => lnum.rem(rnum)?,
            _ => return Ok(vec![]),
        };
        let owned_num = OwnedJsonb::new(Value::Number(num_val).to_vec());
        Ok(vec![JsonbItem::Owned(owned_num)])
    }

    fn eval_value(&mut self, val: &PathValue<'a>) -> Result<JsonbItem<'a>> {
        let value = match val {
            PathValue::Null => Value::Null,
            PathValue::Boolean(v) => Value::Bool(*v),
            PathValue::Number(v) => Value::Number(v.clone()),
            PathValue::String(v) => Value::String(v.clone()),
            PathValue::Raw(v) => {
                return Ok(JsonbItem::Raw(*v));
            }
        };
        // Literals are wrapped in a single-element array so they compare like
        // the array of values selected by the other side of the expression.
        let owned_val = OwnedJsonb::new(Value::Array(vec![value]).to_vec());
        Ok(JsonbItem::Owned(owned_val))
    }

    fn eval_filter_expr(
        &mut self,
        item: JsonbItem<'a>,
        expr: &'a Expr<'a>,
    ) -> Result<Option<bool>> {
        match expr {
            Expr::BinaryOp { op, left, right } => match op {
                BinaryOperator::Or => {
                    let lhs = self.eval_filter_expr(item.clone(), left)?;
                    let rhs = self.eval_filter_expr(item.clone(), right)?;
                    match (lhs, rhs) {
                        (Some(lhs), Some(rhs)) => Ok(Some(lhs || rhs)),
                        (_, _) => Ok(None),
                    }
                }
                BinaryOperator::And => {
                    let lhs = self.eval_filter_expr(item.clone(), left)?;
                    let rhs = self.eval_filter_expr(item.clone(), right)?;
                    match (lhs, rhs) {
                        (Some(lhs), Some(rhs)) => Ok(Some(lhs && rhs)),
                        (_, _) => Ok(None),
                    }
                }
                BinaryOperator::Eq
                | BinaryOperator::NotEq
                | BinaryOperator::Lt
                | BinaryOperator::Lte
                | BinaryOperator::Gt
                | BinaryOperator::Gte
                | BinaryOperator::StartsWith => {
                    let lhs = self.convert_expr_val(item.clone(), left)?;
                    let rhs = self.convert_expr_val(item.clone(), right)?;
                    let res = self.eval_compare(op, &lhs, &rhs);
                    Ok(res)
                }
                _ => Ok(None),
            },
            Expr::ExistsFunc(paths) => {
                let res = self.eval_exists(item, paths)?;
                Ok(Some(res))
            }
            _ => Err(Error::InvalidJsonPath),
        }
    }

    fn eval_exists(&mut self, item: JsonbItem<'a>, paths: &'a [Path<'a>]) -> Result<bool> {
        let filter_items = self.select_by_filter_paths(item, paths)?;
        let res = !filter_items.is_empty();
        Ok(res)
    }

    fn select_by_filter_paths(
        &mut self,
        item: JsonbItem<'a>,
        paths: &'a [Path<'a>],
    ) -> Result<VecDeque<JsonbItem<'a>>> {
        let mut items = VecDeque::new();
        if let Some(Path::Current) = paths.first() {
            items.push_front(item.clone());
        } else {
            let root_item = JsonbItem::Raw(self.root_jsonb);
            items.push_front(root_item);
        }
        std::mem::swap(&mut self.items, &mut items);

        for path in paths.iter() {
            match path {
                &Path::Root | &Path::Current => {
                    continue;
                }
                Path::FilterExpr(expr) => {
                    let len = self.items.len();
                    for _ in 0..len {
                        let item = self.items.pop_front().unwrap();
                        let res = self.eval_filter_expr(item.clone(), expr)?.unwrap_or(false);
                        if res {
                            self.items.push_back(item);
                        }
                    }
                }
                _ => {
                    self.select_by_path(path)?;
                }
            }
        }
        std::mem::swap(&mut self.items, &mut items);
        Ok(items)
    }

    fn convert_expr_val(
        &mut self,
        item: JsonbItem<'a>,
        expr: &'a Expr<'a>,
    ) -> Result<ExprValue<'a>> {
        match expr {
            Expr::Value(value) => Ok(ExprValue::Value(value.clone())),
            Expr::Paths(paths) => {
                let mut filter_items = self.select_by_filter_paths(item, paths)?;

                let mut values = Vec::with_capacity(filter_items.len());
                while let Some(item) = filter_items.pop_front() {
                    let value = match item {
                        JsonbItem::Null => PathValue::Null,
                        JsonbItem::Boolean(v) => PathValue::Boolean(v),
                        JsonbItem::Number(num) => {
                            let n = num.as_number()?;
                            PathValue::Number(n)
                        }
                        JsonbItem::String(s) => PathValue::String(s),
                        JsonbItem::Raw(raw) => {
                            // collect values in the array.
                            let array_iter_opt = ArrayIterator::new(raw)?;
                            if let Some(mut array_iter) = array_iter_opt {
                                for item_result in &mut array_iter {
                                    let item = item_result?;
                                    let value = match item {
                                        JsonbItem::Null => PathValue::Null,
                                        JsonbItem::Boolean(v) => PathValue::Boolean(v),
                                        JsonbItem::Number(num) => {
                                            let n = num.as_number()?;
                                            PathValue::Number(n)
                                        }
                                        JsonbItem::String(s) => PathValue::String(s),
                                        JsonbItem::Raw(raw) => PathValue::Raw(raw),
                                        _ => {
                                            continue;
                                        }
                                    };
                                    values.push(value);
                                }
                            } else {
                                let jsonb_item = JsonbItem::from_raw_jsonb(raw)?;
                                let value = match jsonb_item {
                                    JsonbItem::Null => PathValue::Null,
                                    JsonbItem::Boolean(v) => PathValue::Boolean(v),
                                    JsonbItem::Number(num) => {
                                        let n = num.as_number()?;
                                        PathValue::Number(n)
                                    }
                                    JsonbItem::String(s) => PathValue::String(s),
                                    JsonbItem::Raw(raw) => PathValue::Raw(raw),
                                    _ => {
                                        continue;
                                    }
                                };
                                values.push(value);
                            }
                            continue;
                        }
                        _ => {
                            continue;
                        }
                    };
                    values.push(value);
                }
                Ok(ExprValue::Values(values))
            }
            _ => unreachable!(),
        }
    }

    fn eval_compare(
        &mut self,
        op: &BinaryOperator,
        lhs: &ExprValue<'a>,
        rhs: &ExprValue<'a>,
    ) -> Option<bool> {
        let (lvals, rvals) = match (lhs, rhs) {
            (ExprValue::Value(lhs), ExprValue::Value(rhs)) => {
                (vec![*lhs.clone()], vec![*rhs.clone()])
            }
            (ExprValue::Values(lhses), ExprValue::Value(rhs)) => {
                (lhses.clone(), vec![*rhs.clone()])
            }
            (ExprValue::Value(lhs), ExprValue::Values(rhses)) => {
                (vec![*lhs.clone()], rhses.clone())
            }
            (ExprValue::Values(lhses), ExprValue::Values(rhses)) => (lhses.clone(), rhses.clone()),
        };

        for lval in lvals.iter() {
            for rval in rvals.iter() {
                let res = self.compare_value(op, lval.clone(), rval.clone())?;
                if res {
                    return Some(true);
                }
            }
        }
        Some(false)
    }

    fn compare_value(
        &mut self,
        op: &BinaryOperator,
        lhs: PathValue<'a>,
        rhs: PathValue<'a>,
    ) -> Option<bool> {
        // container value can't compare values.
        if matches!(lhs, PathValue::Raw(_)) || matches!(rhs, PathValue::Raw(_)) {
            return None;
        }
        if op == &BinaryOperator::StartsWith {
            let res = match (lhs, rhs) {
                (PathValue::String(lhs), PathValue::String(rhs)) => Some(lhs.starts_with(&*rhs)),
                (_, _) => None,
            };
            return res;
        }
        let order = lhs.partial_cmp(&rhs);
        if let Some(order) = order {
            let res = match op {
                BinaryOperator::Eq => order == Ordering::Equal,
                BinaryOperator::NotEq => order != Ordering::Equal,
                BinaryOperator::Lt => order == Ordering::Less,
                BinaryOperator::Lte => order == Ordering::Equal || order == Ordering::Less,
                BinaryOperator::Gt => order == Ordering::Greater,
                BinaryOperator::Gte => order == Ordering::Equal || order == Ordering::Greater,
                _ => {
                    return None;
                }
            };
            Some(res)
        } else if matches!(op, BinaryOperator::NotEq) {
            Some(true)
        } else {
            None
        }
    }
}
