# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Opt-in layered IVF_RQ; requires a reader with layered format support."""

import lance


def layered_search(uri, query):
    dataset = lance.dataset(uri)
    dataset.create_index(
        "vector",
        index_type="IVF_RQ",
        num_bits=7,
        layered=True,
        num_partitions=4096,
        metric="cosine",
    )
    return dataset.to_table(
        columns=["_distance"],
        nearest={
            "column": "vector",
            "q": query,
            "k": 1000,
            "nprobes": 64,
            "rq_precision": "full",
            "rq_cascade_factor": 8,
        },
    )
