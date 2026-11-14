// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use crate::{
    buffer::LanceBuffer,
    compression::{BlockDecompressor, levels::LevelDecodePlan},
};

use super::{DefinitionInterpretation, RepDefUnraveler};

fn decode(
    plan: Option<&LevelDecodePlan<'_>>,
    work: &mut Reservation,
) -> Result<Option<ReservedBuffer>> {
    plan.map(|plan| {
        plan.try_decode_in(work)?
            .ok_or_else(|| Error::internal("complete level admission unexpectedly refused"))
    })
    .transpose()
}

impl RepDefUnraveler {
    /// Decode two encoded dense level streams with a joint payload preflight.
    /// The codecs validate framing without expansion. Input owners must already
    /// be charged, and `num_levels` must be the resolved count from the enclosing
    /// page (including any existing legacy count recovery). `num_items` is the
    /// visible leaf count, which can differ from the number of structural levels.
    ///
    /// `None` means insufficient credit and leaves the workspace unchanged.
    /// Invalid decoded levels recycle both outputs before returning an error.
    /// Scratch is reused between streams; decoded buffers transfer directly into
    /// the unraveler without a second copy. This does not schedule input or wire
    /// the ordinary MiniBlock task to admission. See [`LevelDecodePlan`] for the
    /// codec admission example and its payload/metadata boundary.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_buffer::Buffer;
    /// use lance_core::utils::allocation::AllocationBudget;
    /// use lance_encoding::{buffer::LanceBuffer, format::metadata,
    ///     encodings::physical::value::ValueDecompressor,
    ///     repdef::{RepDefUnraveler, CompositeRepDefUnraveler, DefinitionInterpretation}};
    /// let codec = ValueDecompressor::from_flat(&metadata::Flat { bits_per_value: 16 });
    /// let budget = AllocationBudget::new(9); // input + decoded levels + bitmap
    /// let mut work = budget.try_reserve(9).unwrap();
    /// let mut input = work.try_allocate(4).unwrap().unwrap();
    /// input.as_mut().copy_from_slice(bytemuck::cast_slice(&[0_u16, 1]));
    /// let input = LanceBuffer::from(Buffer::from(input.freeze()));
    /// let levels = RepDefUnraveler::try_from_encoded_in(
    ///     None, Some((&codec, &input)), 2,
    ///     Arc::from([DefinitionInterpretation::NullableItem]), 2, &mut work,
    /// ).unwrap().unwrap();
    /// let mut levels = CompositeRepDefUnraveler::new(vec![levels]);
    /// let validity = levels.try_unravel_validity_in(2, &mut work).unwrap().unwrap().unwrap();
    /// assert_eq!(validity.iter().collect::<Vec<_>>(), [true, false]);
    /// drop(input);
    /// drop(levels);
    /// drop(work);
    /// assert_eq!(budget.reserved(), 1);
    /// drop(validity);
    /// assert_eq!(budget.reserved(), 0);
    /// ```
    pub fn try_from_encoded_in(
        rep: Option<(&dyn BlockDecompressor, &LanceBuffer)>,
        def: Option<(&dyn BlockDecompressor, &LanceBuffer)>,
        num_levels: u64,
        meanings: Arc<[DefinitionInterpretation]>,
        num_items: u64,
        workspace: &mut Reservation,
    ) -> Result<Option<Self>> {
        Self::validate_dense_levels(None, None, &meanings)?;
        let rep = rep
            .map(|(codec, data)| codec.plan_levels(data, num_levels))
            .transpose()?;
        let def = def
            .map(|(codec, data)| codec.plan_levels(data, num_levels))
            .transpose()?;
        let rep_output = rep.as_ref().map_or(0, LevelDecodePlan::output_bytes);
        let required = rep_output
            .checked_add(def.as_ref().map_or(0, LevelDecodePlan::workspace_bytes))
            .ok_or_else(|| Error::invalid_input("combined structural workspace overflows"))?
            .max(rep.as_ref().map_or(0, LevelDecodePlan::workspace_bytes));
        if required > workspace.len() {
            return Ok(None);
        }
        let mut rep_output = None;
        let mut def_output = None;
        let result = (|| {
            rep_output = decode(rep.as_ref(), workspace)?;
            def_output = decode(def.as_ref(), workspace)?;
            Self::validate_dense_levels(
                rep_output
                    .as_ref()
                    .map(|b| bytemuck::cast_slice(b.as_ref())),
                def_output
                    .as_ref()
                    .map(|b| bytemuck::cast_slice(b.as_ref())),
                &meanings,
            )
        })();
        if let Err(error) = result {
            if let Some(rep) = rep_output {
                rep.recycle(workspace)?;
            }
            if let Some(def) = def_output {
                def.recycle(workspace)?;
            }
            return Err(error);
        }
        Self::from_admitted_levels(rep_output, def_output, meanings, num_items).map(Some)
    }
}

#[cfg(test)]
mod tests;
