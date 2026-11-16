// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Actual overlay plans and readers composed inside an admitted fragment take.

use lance_file::reader::PreparedFullZipTake;

use super::*;

struct PreparedField {
    column: usize,
    name: String,
    routing: AdmittedOverlayRouting,
    inputs: Vec<PreparedFullZipTake>,
}

pub(crate) struct PreparedOverlayBatch {
    fields: Vec<PreparedField>,
}

impl PreparedOverlayBatch {
    pub(crate) async fn try_prepare_in(
        selection: &OverlayBatchSelection,
        plans: &[LoadedAtomicField],
        schema: &arrow_schema::Schema,
        workspace: &mut Reservation,
    ) -> Result<Option<Self>> {
        // The supported leaf/list path has one plan per top-level field. General
        // struct reads must carry ancestor validity through admitted decoding;
        // rejecting here prevents an accidental ordinary fetch/splice fallback.
        for plan in plans {
            if schema.index_of(&plan.top_field.name).is_ok()
                && (!plan.ancestor_ids.is_empty() || !plan.struct_ancestor_ids.is_empty())
            {
                return Err(Error::not_supported(
                    "admitted fragment overlays do not yet decode struct ancestor validity",
                ));
            }
        }
        let mut fields = Vec::with_capacity(plans.len());
        for plan in plans {
            let Ok(column) = schema.index_of(&plan.top_field.name) else {
                continue;
            };
            if !plan.is_value_projected {
                continue;
            }
            let coverages: Vec<&RoaringBitmap> = plan
                .overlays_newest_first
                .iter()
                .map(|overlay| overlay.coverage.as_ref())
                .collect();
            if !selected_rows(selection)
                .any(|row| coverages.iter().any(|coverage| coverage.contains(row)))
            {
                continue;
            }
            let Some(routing) = AdmittedOverlayRouting::try_new(selection, &coverages, workspace)?
            else {
                return Ok(None);
            };
            let mut inputs = Vec::with_capacity(plan.overlays_newest_first.len());
            for (index, overlay) in plan.overlays_newest_first.iter().enumerate() {
                let offsets = routing.fetch_offsets(index)?;
                let Some(input) = overlay
                    .reader
                    .prepare_admitted_fullzip_take(
                        &offsets,
                        plan.fetch_projection.clone(),
                        workspace,
                    )
                    .await?
                else {
                    return Ok(None);
                };
                inputs.push(input);
            }
            fields.push(PreparedField {
                column,
                name: plan.top_field.name.clone(),
                routing,
                inputs,
            });
        }
        Ok(Some(Self { fields }))
    }

    pub(crate) fn decode_and_merge_in(
        self,
        base: RecordBatch,
        workspace: &mut Reservation,
    ) -> Result<Option<RecordBatch>> {
        let schema = base.schema();
        let mut columns = base.columns().to_vec();
        drop(base);
        for field in self.fields {
            let mut fetched = Vec::with_capacity(field.inputs.len());
            for input in field.inputs {
                let Some(batch) = input.decode_in(workspace)? else {
                    return Ok(None);
                };
                fetched.push(
                    batch
                        .column_by_name(&field.name)
                        .ok_or_else(|| {
                            Error::invalid_input(
                                "admitted overlay input is missing its projected field",
                            )
                        })?
                        .clone(),
                );
            }
            let Some(output) =
                field
                    .routing
                    .try_interleave_in(&columns[field.column], &fetched, workspace)?
            else {
                return Ok(None);
            };
            columns[field.column] = output;
        }
        Ok(Some(RecordBatch::try_new(schema, columns)?))
    }
}
