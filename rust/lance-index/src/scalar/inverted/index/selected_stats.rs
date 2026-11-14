// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// The row domain is independent of query terms and candidate filters. Retain
/// only the count and DocId mapping, never another membership copy or the
/// complete address/length projection.
#[derive(Debug, DeepSizeOf)]
pub(super) struct SelectedRowDomain {
    total_tokens: u64,
    num_rows: usize,
    doc_ids: Box<[u32]>,
    doc_addresses: Box<[u64]>,
}

impl SelectedRowDomain {
    fn new(docs: &AddressKeyedDocuments, selection: &RowAddrTreeMap) -> Result<Self> {
        let mut rows = (0..docs.len())
            .map(|doc| docs.row_address(doc as u32))
            .filter(|row| *row != RowAddress::TOMBSTONE_ROW && selection.contains(*row))
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows.dedup();
        let mut total_tokens = 0;
        rows.retain(|row| {
            let length = docs.doc_length_at(*row);
            total_tokens += length;
            length != 0
        });
        let mut pairs = Vec::with_capacity(rows.len());
        for row in &rows {
            let doc_ids = docs.doc_ids_at(*row).ok_or_else(|| {
                Error::invalid_input("Selected row domains require DocId-keyed partitions")
            })?;
            pairs.extend(doc_ids.map(|doc_id| (doc_id, *row)));
        }
        pairs.sort_unstable();
        let (doc_ids, doc_addresses): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Ok(Self {
            total_tokens,
            num_rows: rows.len(),
            doc_ids: doc_ids.into_boxed_slice(),
            doc_addresses: doc_addresses.into_boxed_slice(),
        })
    }
}

impl InvertedPartition {
    pub(super) async fn selected_domain_stats_for_terms(
        &self,
        terms: &[String],
        selection: &RowAddrTreeMap,
        serialized_selection: &[u8],
        metrics: Option<&dyn MetricsCollector>,
    ) -> Result<(u64, usize, Vec<usize>)> {
        let domain = self
            .inverted_list
            .index_cache
            .get_or_insert_with_key(
                SelectedRowDomainKey {
                    selection: serialized_selection.to_vec(),
                },
                || async {
                    let docs = self.docs.address_keyed().await?;
                    SelectedRowDomain::new(&docs, selection)
                },
            )
            .await?;
        let mut frequencies = vec![0; terms.len()];
        if domain.num_rows == 0 {
            return Ok((0, 0, frequencies));
        }
        for (slot, term) in terms.iter().enumerate() {
            let Some(token_id) = self.tokens.get(term) else {
                continue;
            };
            let posting = self
                .inverted_list
                .posting_list(token_id, false, metrics.unwrap_or(&NoOpMetricsCollector))
                .await?;
            let mut matches = RoaringTreemap::new();
            match &posting {
                PostingList::Compressed(posting) => {
                    for (row, contained) in domain
                        .doc_addresses
                        .iter()
                        .zip(posting.contains_each(&domain.doc_ids))
                    {
                        if contained {
                            matches.insert(*row);
                        }
                    }
                }
                _ => {
                    let docs = self.docs.address_keyed().await?;
                    for (row, _) in live_posting_rows(&posting, &docs, false) {
                        // The exact selection is already available; retaining
                        // its selected addresses again would compete with postings.
                        if selection.contains(row) && docs.doc_length_at(row) != 0 {
                            matches.insert(row);
                        }
                    }
                }
            }
            frequencies[slot] = matches.len() as usize;
        }
        Ok((domain.total_tokens, domain.num_rows, frequencies))
    }
}
