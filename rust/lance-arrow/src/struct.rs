// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Extension to arrow struct arrays

use arrow_array::{Array, StructArray, cast::AsArray, make_array};
use arrow_data::{ArrayData, ArrayDataBuilder};
use arrow_schema::ArrowError;

use crate::compose_child_validity;

pub trait StructArrayExt {
    /// Removes the offset / length of the struct array by pushing it into the children
    ///
    /// In arrow-rs when slice is called it recursively slices the children.
    /// In arrow-cpp when slice is called it just sets the offset/length of
    ///   the struct array and leaves the children as-is
    ///
    /// Both are legal approaches (╥﹏╥)
    ///
    /// This method helps reduce complexity by folding into the arrow-rs approach
    fn normalize_slicing(&self) -> Result<Self, ArrowError>
    where
        Self: Sized;

    /// Structs are allowed to mask valid items.  For example, a struct array might be:
    ///
    /// [ {"items": [1, 2, 3]}, NULL, {"items": NULL}]
    ///
    /// However, the underlying items array might be: [[1, 2, 3], [4, 5], NULL]
    ///
    /// The [4, 5] list is masked out because the struct array is null.
    ///
    /// The struct validity would be [true, false, true] and the list validity would be [true, true, false]
    ///
    /// This method pushes nulls down into all children.  In the above example the list validity would become
    /// [true, false, false].
    ///
    /// This method is not recursive.  If a child is a struct array it will not push that child's nulls down.
    ///
    /// This method does not remove garbage lists.  It only updates the validity so a future call to
    /// [crate::list::ListArrayExt::filter_garbage_nulls] will remove the garbage lists (without
    /// this pushdown it would not)
    fn pushdown_nulls(&self) -> Result<Self, ArrowError>
    where
        Self: Sized;
}

fn normalized_struct_array_data(data: ArrayData) -> Result<ArrayData, ArrowError> {
    let parent_offset = data.offset();
    let parent_len = data.len();
    let modified_children = data
        .child_data()
        .iter()
        .map(|d| {
            let d = normalized_struct_array_data(d.clone())?;
            let offset = d.offset();
            let len = d.len();
            if len < parent_len + parent_offset {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Child array {} has length {} which is less than the parent length {} plus the parent offset {}",
                    d.data_type(),
                    len,
                    parent_len,
                    parent_offset
                )));
            }
            let new_offset = offset + parent_offset;
            d.into_builder().offset(new_offset)
                .len(parent_len)
                .build()
        })
        .collect::<Result<Vec<_>, _>>()?;
    ArrayDataBuilder::new(data.data_type().clone())
        .len(parent_len)
        .offset(0)
        .buffers(data.buffers().to_vec())
        .child_data(modified_children)
        .build()
}

impl StructArrayExt for StructArray {
    fn normalize_slicing(&self) -> Result<Self, ArrowError>
    where
        Self: Sized,
    {
        if self.offset() == 0 && self.columns().iter().all(|c| c.len() == self.len()) {
            return Ok(self.clone());
        }

        let data = normalized_struct_array_data(self.to_data())?;
        Ok(Self::from(data))
    }

    fn pushdown_nulls(&self) -> Result<Self, ArrowError>
    where
        Self: Sized,
    {
        let Some(validity) = self.nulls() else {
            return Ok(self.clone());
        };
        let children = self
            .columns()
            .iter()
            .map(|child| {
                child
                    .to_data()
                    .into_builder()
                    .nulls(compose_child_validity(child.as_ref(), Some(validity)))
                    .build()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let arr = make_array(self.to_data().into_builder().child_data(children).build()?);
        Ok(arr.as_struct().clone())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        Array, ArrayRef, DictionaryArray, Int32Array, NullArray, StringArray, StructArray,
        cast::AsArray, make_array, types::Int32Type,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::{DataType, Field, Fields};
    use std::sync::Arc;

    use crate::r#struct::StructArrayExt;

    #[test]
    fn test_normalize_slicing_no_offset() {
        let x = Int32Array::from(vec![1, 2, 3]);
        let y = Int32Array::from(vec![4, 5, 6]);
        let struct_array = StructArray::new(
            Fields::from(vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, true),
            ]),
            vec![Arc::new(x), Arc::new(y)],
            None,
        );

        let normalized = struct_array.normalize_slicing().unwrap();
        assert_eq!(normalized, struct_array);
    }

    #[test]
    fn test_arrow_rs_slicing() {
        let x = Int32Array::from(vec![1, 2, 3, 4]);
        let y = Int32Array::from(vec![5, 6, 7, 8]);
        let struct_array = StructArray::new(
            Fields::from(vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, true),
            ]),
            vec![Arc::new(x), Arc::new(y)],
            None,
        );

        // Slicing with arrow-rs propagates the slicing to the children so there should
        // be no change needed to the struct array
        let sliced = struct_array.slice(1, 2);
        let normalized = sliced.normalize_slicing().unwrap();

        assert_eq!(normalized, sliced);
    }

    #[test]
    fn test_arrow_cpp_slicing() {
        let x = Int32Array::from(vec![1, 2, 3, 4]);
        let y = Int32Array::from(vec![5, 6, 7, 8]);
        let struct_array = StructArray::new(
            Fields::from(vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, true),
            ]),
            vec![Arc::new(x), Arc::new(y)],
            None,
        );

        let data = struct_array.to_data();
        let sliced = data.into_builder().offset(1).len(2).build().unwrap();
        let sliced = make_array(sliced);
        let normalized = sliced.as_struct().clone().normalize_slicing().unwrap();

        assert_eq!(normalized, struct_array.slice(1, 2));
    }

    #[rstest::rstest]
    #[case::same_offsets(0, 0)]
    #[case::parent_bit_offset(0, 9)]
    #[case::child_bit_offset(9, 0)]
    #[case::different_offsets(7, 9)]
    #[case::large_parent_offset(1, 1025)]
    #[case::large_child_offset(1025, 1)]
    fn test_pushdown_nulls_with_null_typed_child_and_sliced_validity(
        #[case] child_offset: usize,
        #[case] parent_offset: usize,
        #[values(false, true)] child_has_nulls: bool,
    ) {
        // `pa.Table.from_pylist([{"s": {"a": 1, "b": None}}, {"s": None}])` infers a
        // struct with a Null-typed child. Arrow forbids a validity buffer on a Null
        // array, so the struct's nulls must not be pushed into it.
        let mut values = vec![Some(-1); child_offset];
        values.extend([
            Some(1),
            Some(2),
            if child_has_nulls { None } else { Some(3) },
            Some(4),
        ]);
        let child: ArrayRef = Arc::new(Int32Array::from(values).slice(child_offset, 4));
        let mut parent_validity = vec![true; parent_offset];
        parent_validity.extend([true, false, true, true]);
        let parent_validity = NullBuffer::from(parent_validity).slice(parent_offset, 4);

        let struct_array = StructArray::new(
            Fields::from(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Null, true),
            ]),
            vec![child, Arc::new(NullArray::new(4))],
            Some(parent_validity),
        );

        let pushed = struct_array.pushdown_nulls().unwrap();
        pushed.to_data().validate_full().unwrap();

        assert_eq!(pushed.nulls(), struct_array.nulls());
        assert_eq!(
            pushed
                .column(0)
                .as_primitive::<Int32Type>()
                .iter()
                .collect::<Vec<_>>(),
            vec![
                Some(1),
                None,
                if child_has_nulls { None } else { Some(3) },
                Some(4)
            ]
        );
        assert_eq!(pushed.column(1).data_type(), &DataType::Null);
        assert!(pushed.column(1).nulls().is_none());
        assert_eq!(pushed.column(1).logical_null_count(), 4);
    }

    #[test]
    fn test_pushdown_nulls_reuses_sufficient_parent_validity_buffer() {
        let child: ArrayRef = Arc::new(Int32Array::from(vec![-1, 1, 2, 3, 4]).slice(1, 4));
        let mut parent_validity = vec![true; 1024];
        parent_validity[2] = false;
        let parent_validity = NullBuffer::from(parent_validity).slice(1, 4);
        let original_buffer = parent_validity.buffer().as_ptr();
        let struct_array = StructArray::new(
            Fields::from(vec![Field::new("a", DataType::Int32, true)]),
            vec![child],
            Some(parent_validity),
        );

        let pushed = struct_array.pushdown_nulls().unwrap();

        pushed.to_data().validate_full().unwrap();
        assert_eq!(
            pushed.column(0).nulls().unwrap().buffer().as_ptr(),
            original_buffer
        );
    }

    #[test]
    fn test_pushdown_nulls_preserves_null_dictionary_values() {
        let values: ArrayRef = Arc::new(StringArray::from(vec![Some("zero"), None, Some("two")]));
        let dictionary: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::new(
            Int32Array::from(vec![0, 1, 2, 0]),
            values.clone(),
        ));
        let struct_array = StructArray::new(
            Fields::from(vec![Field::new(
                "dict",
                dictionary.data_type().clone(),
                true,
            )]),
            vec![dictionary],
            Some(NullBuffer::from(vec![true, true, false, true])),
        );

        let pushed = struct_array.pushdown_nulls().unwrap();
        let pushed = pushed
            .column(0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap();

        assert_eq!(pushed.values(), &values);
        assert!(pushed.values().is_null(1));
        assert!(pushed.keys().is_valid(1));
        assert!(pushed.keys().is_null(2));
    }
}
