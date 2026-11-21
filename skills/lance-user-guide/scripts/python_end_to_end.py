#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa

import lance


def _build_fixed_size_vectors(
    num_rows: int, dim: int
) -> tuple[pa.FixedSizeListArray, np.ndarray]:
    vectors = np.random.rand(num_rows, dim).astype("float32")
    flat = pa.array(vectors.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, dim), vectors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal Lance write/index/query example"
    )
    parser.add_argument(
        "--uri", default="example.lance", help="Dataset URI (directory)"
    )
    parser.add_argument(
        "--mode", default="overwrite", choices=["create", "append", "overwrite"]
    )
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=32)

    parser.add_argument("--build-scalar-index", action="store_true")
    parser.add_argument("--build-vector-index", action="store_true")

    parser.add_argument("--vector-index-type", default="IVF_PQ")
    parser.add_argument("--target-partition-size", type=int, default=8192)
    parser.add_argument("--num-sub-vectors", type=int, default=8)

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--filter", default="category = 'a'")
    parser.add_argument("--prefilter", action="store_true")

    args = parser.parse_args()

    uri = str(Path(args.uri))
    vec_arr, vec_np = _build_fixed_size_vectors(args.rows, args.dim)
    categories = pa.array(["a" if i % 2 == 0 else "b" for i in range(args.rows)])
    table = pa.table(
        {
            "id": pa.array(range(args.rows), pa.int64()),
            "category": categories,
            "vector": vec_arr,
        }
    )

    ds = lance.write_dataset(table, uri, mode=args.mode)
    ds = lance.dataset(uri)

    if args.build_scalar_index:
        ds.create_scalar_index("category", "BTREE", replace=True)

    if args.build_vector_index:
        ds = ds.create_index(
            "vector",
            index_type=args.vector_index_type,
            target_partition_size=args.target_partition_size,
            num_sub_vectors=args.num_sub_vectors,
        )

    print(f"uri={ds.uri}")
    print(f"rows={ds.count_rows()}")
    print("indices=")
    for idx in ds.describe_indices():
        print(f"  - {idx}")

    q = vec_np[0]
    scan = ds.scanner(
        nearest={"column": "vector", "q": q, "k": args.k},
        filter=args.filter if args.filter else None,
        prefilter=args.prefilter,
    )
    result = scan.to_table()
    print("result_schema=")
    print(result.schema)
    print("result_preview=")
    print(result.slice(0, 5).to_pydict())


if __name__ == "__main__":
    main()
