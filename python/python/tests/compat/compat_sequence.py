# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Cross-version index maintenance-sequence search.

Runs on the same per-ref venv substrate as the rest of this package: venv_factory
(venv_manager.py) provisions one venv per ref, so the *setup* half of a sequence runs
under `from_ref` and the *exercise* half under `to_ref` (the version split). After each
run an oracle checks that the reader did not panic and that an index query agrees with a
full (unindexed) scan. This *discovers* cross-version regressions (e.g. ENT-1662)
without hand-coding the triggering sequence.

The scenario is parameterized by index *kind* so every scalar index type gets the same
aged-lifecycle, cross-version treatment. The scalar oracle runs the same predicate twice
-- normally and with use_scalar_index=False (lance ignores the index) -- and requires
the results to match. If the two query plans are identical the index wasn't used, so the
comparison is skipped rather than failed (uninformative, not a regression). FTS has no
"ignore the index" mode to diff against, so its oracle reconstructs ground truth from a
full scan: tokenize every live row, then require an FTS search for a spread of sampled
terms to return exactly the rows that contain them. The FTS scenarios run under both
on-disk format versions (1 and 2), which take different merge paths. New Lance versions
pin this with the create-index `format_version` parameter; old Lance versions still use
`LANCE_FTS_FORMAT_VERSION`.

IVF_PQ uses a separate bounded grammar because exhaustively applying the scalar grammar
to a trained vector index would be prohibitively expensive. It covers every ordered pair
of vector/scalar maintenance operations plus a few deeper lifecycle sequences. Its
oracle compares scalar-prefiltered ANN results with an exact, index-free KNN scan and
requires at least 0.5 recall, as well as checking the scalar filter exactly.

The op vocabulary and bounds are deliberately small so the search is runnable. Scalar
and FTS cases are exhaustive over their maintenance grammar up to the configured length;
vector cases cover every ordered pair plus the curated deeper lifecycles above.
"""

import itertools
import os
import shutil
from pathlib import Path

ROWS_PER_WRITE = 200
VECTOR_ROWS_PER_WRITE = 512
VECTOR_DIM = 8
VECTOR_K = 10
VECTOR_KIND = "IVF_PQ"
VECTOR_INDEX_NAME = "vector_idx"
VECTOR_SCALAR_INDEX_NAME = "scalar_idx"

SETUP_TAIL_OPS = ["D", "C", "W"]
EXERCISE_OPS = ["W", "D", "C", "Oa", "Om", "Od"]
VECTOR_OPS = ("W", "D", "C", "Os", "Ov", "Om")

# All ordered operation pairs are searched below. These longer cases preserve the
# state combinations that motivated the old recurring test without growing as 6**N.
VECTOR_CRITICAL_SEQUENCES = (
    ("W", "Ov", "W", "Ov", "W"),  # two vector deltas, then unindexed rows
    ("W", "Os", "W", "Ov", "Om"),
    ("D", "C", "W", "Ov", "Om"),
    ("W", "D", "Os", "C", "Ov"),
    ("W", "Ov", "D", "C", "Om"),
)

OP_NAMES = {
    "W": "write rows",
    "I": "create index",
    "D": "delete rows",
    "C": "compact",
    "Oa": "optimize (append)",
    "Om": "optimize (merge)",
    "Od": "optimize",
    "Os": "optimize scalar index (append)",
    "Ov": "optimize vector index (append)",
}


def describe(kind, from_ref, to_ref, setup_ops, exercise_ops, fts_version=None):
    """A plain-English description of a scenario for failure output."""
    writer = ", then ".join(OP_NAMES[o] for o in ["W", "I", *setup_ops])
    reader = ", then ".join(OP_NAMES[o] for o in exercise_ops)
    tag = f" (fts fmt v{fts_version})" if fts_version is not None else ""
    return f"{kind}{tag} ({from_ref} -> {to_ref}): writer [{writer}]; reader [{reader}]"


# Index kinds covered by the maintenance-sequence search.
SCALAR_KINDS = ["BTREE", "BITMAP", "LABEL_LIST", "NGRAM", "ZONEMAP", "BLOOMFILTER"]
ALL_KINDS = ["INVERTED", *SCALAR_KINDS, VECTOR_KIND]


class IndexScenario:
    """A picklable, kind-parameterized scenario run across a version split."""

    def __init__(self, kind, path, setup_ops, exercise_ops, fts_version=None):
        self.kind = kind
        self.path = str(path)
        self.setup_ops = list(setup_ops)
        self.exercise_ops = list(exercise_ops)
        self.fts_version = fts_version
        self.next_idx = 0

    # --- in-venv helpers (only lance + pyarrow available) ---
    def _open(self):
        import lance

        session = lance.Session(index_cache_size_bytes=0, metadata_cache_size_bytes=0)
        return lance.dataset(self.path, session=session)

    def _batch(self, a, b):
        import pyarrow as pa

        idx = list(range(a, b))
        if self.kind == VECTOR_KIND:
            # Deterministic pseudo-random vectors keep every appended batch in the
            # training distribution without depending on numpy or process RNG state.
            flat = []
            for i in idx:
                state = ((i + 1) * 2654435761) & 0xFFFFFFFF
                for _ in range(VECTOR_DIM):
                    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
                    flat.append(state / 4294967296.0)
            vector = pa.FixedSizeListArray.from_arrays(
                pa.array(flat, type=pa.float32()), VECTOR_DIM
            )
            return pa.table({"idx": idx, "vector": vector})
        if self.kind == "INVERTED":
            # Each row's text mixes tokens of different frequency: a unique term, a
            # mid-frequency bucket (~1/7 of rows), and one shared by every row. Sampling
            # across that spread exercises postings of varied length.
            return pa.table(
                {"idx": idx, "key": [f"term{i} bucket{i % 7} shared" for i in idx]}
            )
        if self.kind == "LABEL_LIST":
            return pa.table({"idx": idx, "key": [[f"l{i % 8}"] for i in idx]})
        if self.kind == "NGRAM":
            return pa.table({"idx": idx, "key": [f"w{i % 50}x" for i in idx]})
        # BTREE / BITMAP / ZONEMAP / BLOOMFILTER: integer column
        card = 8 if self.kind == "BITMAP" else 50
        key = [i if self.kind == "ZONEMAP" else i % card for i in idx]
        return pa.table({"idx": idx, "key": key})

    def _index_type(self):
        return "INVERTED" if self.kind == "INVERTED" else self.kind

    def _oracle_pred(self):
        if self.kind == "LABEL_LIST":
            return "array_has_any(key, ['l3'])"
        if self.kind == "NGRAM":
            return "contains(key, 'w3x')"
        if self.kind == "ZONEMAP":
            return "key >= 100 AND key < 300"
        return "key == 3"  # BTREE / BITMAP / BLOOMFILTER

    # --- ops ---
    def _op_W(self):
        import lance

        num_rows = VECTOR_ROWS_PER_WRITE if self.kind == VECTOR_KIND else ROWS_PER_WRITE
        a, b = self.next_idx, self.next_idx + num_rows
        self.next_idx = b
        tbl = self._batch(a, b)
        if not os.path.exists(self.path):
            lance.write_dataset(tbl, self.path)  # single fragment
        else:
            self._open().insert(tbl)

    def _op_I(self):
        if self.kind == VECTOR_KIND:
            self._open().create_scalar_index(
                "idx", "BTREE", name=VECTOR_SCALAR_INDEX_NAME
            )
            self._open().create_index(
                "vector",
                index_type="IVF_PQ",
                name=VECTOR_INDEX_NAME,
                num_partitions=2,
                num_sub_vectors=2,
            )
            return
        kwargs = {"with_position": True} if self.kind == "INVERTED" else {}
        if self.kind == "INVERTED" and self.fts_version is not None:
            kwargs["format_version"] = int(self.fts_version)
        self._open().create_scalar_index("key", self._index_type(), **kwargs)

    def _op_D(self):
        # Partial-range delete inside the id space so compaction rewrites and remaps the
        # index per-row.
        if self.next_idx == 0:
            return
        lo, hi = self.next_idx // 4, self.next_idx // 2
        if hi > lo:
            self._open().delete(f"idx >= {lo} AND idx < {hi}")

    def _op_C(self):
        self._open().optimize.compact_files()

    def _op_Oa(self):
        self._open().optimize.optimize_indices(num_indices_to_merge=0)

    def _op_Om(self):
        kwargs = {"num_indices_to_merge": 10}
        if self.kind == VECTOR_KIND:
            kwargs = {
                "num_indices_to_merge": 1,
                "index_names": [VECTOR_INDEX_NAME],
            }
        self._open().optimize.optimize_indices(**kwargs)

    def _op_Od(self):
        self._open().optimize.optimize_indices()

    def _op_Os(self):
        self._open().optimize.optimize_indices(
            num_indices_to_merge=0, index_names=[VECTOR_SCALAR_INDEX_NAME]
        )

    def _op_Ov(self):
        self._open().optimize.optimize_indices(
            num_indices_to_merge=0, index_names=[VECTOR_INDEX_NAME]
        )

    def _run(self, ops):
        for op in ops:
            getattr(self, f"_op_{op}")()

    # --- methods invoked across the version split ---
    def setup(self):
        shutil.rmtree(self.path, ignore_errors=True)
        self.next_idx = 0
        self._run(["W", "I"] + self.setup_ops)
        return self.next_idx

    def exercise_and_check(self):
        self._run(self.exercise_ops)
        ds = self._open()
        if self.kind == VECTOR_KIND:
            self._check_vector_prefilter(ds)
            return
        if self.kind == "INVERTED":
            # Differential oracle: rebuild the token -> rows map from a full (unindexed)
            # scan, then require an FTS search for a spread of sampled terms to return
            # exactly those rows. Catches a merge that drops or misassigns postings, not
            # just a row-count drift. (Tokens here are alphanumeric and space-separated,
            # so a whitespace split reproduces lance's tokenization.)
            rows = ds.to_table(columns=["idx", "key"])
            idxs = rows.column("idx").to_pylist()
            texts = rows.column("key").to_pylist()
            truth = {}
            for i, text in zip(idxs, texts):
                for tok in text.split():
                    truth.setdefault(tok, set()).add(i)
            if not truth:
                return  # everything deleted; nothing to search
            vocab = sorted(truth)
            # A spread across the vocabulary plus the most common term.
            sample = set(vocab[:: max(1, len(vocab) // 6)])
            sample.add(max(truth, key=lambda t: len(truth[t])))
            for term in sorted(sample):
                hit = ds.to_table(full_text_query={"query": term, "columns": ["key"]})
                got = set(hit.column("idx").to_pylist())
                want = truth[term]
                assert got == want, (
                    f"FTS('{term}'): index returned {len(got)} rows, corpus has "
                    f"{len(want)} (missing {sorted(want - got)[:5]}, "
                    f"extra {sorted(got - want)[:5]})"
                )
            return
        # Same column/predicate, index on vs forced off: use_scalar_index=False makes
        # lance ignore the index, so the plans differ iff the index is used. If they are
        # identical the index wasn't consulted here (the planner chose a scan after
        # deletes), so the comparison is vacuous -- skip rather than compare two scans.
        pred = self._oracle_pred()
        plan_index = ds.scanner(filter=pred).explain_plan(True)
        plan_scan = ds.scanner(filter=pred, use_scalar_index=False).explain_plan(True)
        if plan_index == plan_scan:
            return
        got = ds.to_table(filter=pred).num_rows
        expected = ds.to_table(filter=pred, use_scalar_index=False).num_rows
        assert got == expected, (
            f"{self.kind}: index gave {got} rows, full scan {expected}, for '{pred}'"
        )

    def _check_vector_prefilter(self, ds):
        """Check both BTREE filtering and IVF_PQ recall against index-free scans."""
        # Both ranges avoid the deterministic delete window. The first is always in
        # the original index while the newest range may be an unindexed append, so
        # the same query covers the indexed + unindexed prefilter path.
        filter_rows = VECTOR_ROWS_PER_WRITE // 8
        lo = self.next_idx - filter_rows
        pred = f"idx < {filter_rows} OR (idx >= {lo} AND idx < {self.next_idx})"

        filtered_scan = ds.to_table(
            columns=["idx", "vector"], filter=pred, use_scalar_index=False
        )
        filtered_rows = sorted(
            zip(
                filtered_scan.column("idx").to_pylist(),
                filtered_scan.column("vector").to_pylist(),
            )
        )
        filtered_ids = [idx for idx, _ in filtered_rows]
        filtered_id_set = set(filtered_ids)
        assert len(filtered_ids) == len(filtered_id_set), (
            f"BTREE prefilter returned duplicate row ids for '{pred}'"
        )
        assert len(filtered_ids) >= VECTOR_K, (
            f"not enough live rows ({len(filtered_ids)}) for vector oracle '{pred}'"
        )

        scalar_plan = ds.scanner(filter=pred).explain_plan(True)
        scan_plan = ds.scanner(filter=pred, use_scalar_index=False).explain_plan(True)
        scalar_markers = ("ScalarIndexQuery", "MaterializeIndex")
        assert any(marker in scalar_plan for marker in scalar_markers), (
            f"BTREE index was not used for '{pred}':\n{scalar_plan}"
        )
        assert not any(marker in scan_plan for marker in scalar_markers), (
            f"BTREE index disabling was ignored for '{pred}':\n{scan_plan}"
        )
        scalar_ids = ds.to_table(columns=["idx"], filter=pred).column("idx").to_pylist()
        assert len(scalar_ids) == len(set(scalar_ids)), (
            f"BTREE returned duplicate row ids for '{pred}'"
        )
        assert set(scalar_ids) == filtered_id_set, (
            f"BTREE returned {len(scalar_ids)} rows, full scan returned "
            f"{len(filtered_ids)}, for '{pred}'"
        )

        query_positions = (0, len(filtered_ids) // 2, len(filtered_ids) - 1)
        vectors = [vector for _, vector in filtered_rows]
        for query_position in query_positions:
            query = vectors[query_position]
            indexed_nearest = {
                "column": "vector",
                "q": query,
                "k": VECTOR_K,
                "nprobes": 2,
                "refine_factor": 10,
            }
            exact_nearest = {
                "column": "vector",
                "q": query,
                "k": VECTOR_K,
                "use_index": False,
            }

            ann_plan = ds.scanner(
                nearest=indexed_nearest, filter=pred, prefilter=True
            ).explain_plan(True)
            exact_plan = ds.scanner(
                nearest=exact_nearest,
                filter=pred,
                prefilter=True,
                use_scalar_index=False,
            ).explain_plan(True)
            ann_markers = ("ANNSubIndex", "ANNIvfPartition")
            assert any(marker in ann_plan for marker in ann_markers), (
                f"IVF_PQ index was not used by vector search:\n{ann_plan}"
            )
            assert not any(marker in exact_plan for marker in ann_markers), (
                f"IVF_PQ index disabling was ignored:\n{exact_plan}"
            )
            assert any(marker in ann_plan for marker in scalar_markers), (
                f"BTREE index was not used by vector prefilter:\n{ann_plan}"
            )
            assert not any(marker in exact_plan for marker in scalar_markers), (
                f"BTREE index disabling was ignored by exact prefilter:\n{exact_plan}"
            )

            got = (
                ds.to_table(
                    columns=["idx", "_distance"],
                    nearest=indexed_nearest,
                    filter=pred,
                    prefilter=True,
                )
                .column("idx")
                .to_pylist()
            )
            expected = (
                ds.to_table(
                    columns=["idx", "_distance"],
                    nearest=exact_nearest,
                    filter=pred,
                    prefilter=True,
                    use_scalar_index=False,
                )
                .column("idx")
                .to_pylist()
            )

            assert len(got) == len(set(got)), "IVF_PQ search returned duplicate row ids"
            assert len(expected) == len(set(expected)), (
                "exact vector search returned duplicate row ids"
            )
            assert set(got) <= filtered_id_set, (
                f"IVF_PQ prefilter returned ids outside '{pred}': "
                f"{sorted(set(got) - filtered_id_set)[:5]}"
            )
            assert set(expected) <= filtered_id_set, (
                f"exact prefilter returned ids outside '{pred}': "
                f"{sorted(set(expected) - filtered_id_set)[:5]}"
            )
            assert len(got) == len(expected) == VECTOR_K, (
                f"IVF_PQ returned {len(got)} rows, exact search returned "
                f"{len(expected)}"
            )
            recall = len(set(got) & set(expected)) / VECTOR_K
            assert recall >= 0.5, (
                f"IVF_PQ prefilter recall@{VECTOR_K}={recall:.3f}; "
                f"expected at least 0.5 (got={got}, exact={expected})"
            )


def generate(max_length):
    """Yield every (setup_ops, exercise_ops) whose combined length is 1..max_length,
    breadth-first by total length (shorter first). `max_length` is the number of
    maintenance ops after the implicit write + create-index, split between the writer
    (setup) and reader (exercise) at every position. The order is neutral, so finding a
    bug is a real search, not a sorted shortcut. The space grows fast with max_length,
    so deeper bugs (ENT-1662 needs length 5) cost more to reach."""
    for total in range(1, max_length + 1):
        for setup_len in range(total):  # exercise gets total - setup_len >= 1
            for s in itertools.product(SETUP_TAIL_OPS, repeat=setup_len):
                for e in itertools.product(EXERCISE_OPS, repeat=total - setup_len):
                    yield list(s), list(e)


def generate_vector(max_length):
    """Yield a bounded IVF_PQ + BTREE maintenance search space.

    Every ordered pair of operations is covered on both sides of the version split.
    A small set of deeper cases captures multi-delta, unindexed-row, delete, compact,
    and merge interactions without expanding the full operation grammar to 6**N.
    """
    seen = set()
    for total in range(1, min(max_length, 2) + 1):
        for sequence in itertools.product(VECTOR_OPS, repeat=total):
            for setup_len in range(total):
                case = (sequence[:setup_len], sequence[setup_len:])
                if case not in seen:
                    seen.add(case)
                    yield list(case[0]), list(case[1])
    for sequence in VECTOR_CRITICAL_SEQUENCES:
        if len(sequence) > max_length:
            continue
        for setup_len in range(len(sequence)):
            case = (sequence[:setup_len], sequence[setup_len:])
            if case not in seen:
                seen.add(case)
                yield list(case[0]), list(case[1])


def search(
    venv_factory,
    from_ref,
    to_ref,
    base_path,
    kind,
    max_length=4,
    shard=0,
    num_shards=1,
    stop_on_first=True,
    fts_version=None,
):
    """Search index-maintenance sequences up to `max_length` ops for one `kind`, across
    (from_ref -> to_ref). Runs only scenarios in this shard (i % num_shards == shard) so
    the space can be split across parallel workers. For INVERTED, `fts_version` ("1" or
    "2") pins the on-disk FTS format on both sides. New Lance versions receive this
    through the create-index parameter and old Lance versions receive it through
    LANCE_FTS_FORMAT_VERSION. Both are Fst token sets and exercise distinct merge paths.
    Returns failures; stops on the first when `stop_on_first`."""
    from_venv = venv_factory.get_venv(from_ref)
    to_venv = venv_factory.get_venv(to_ref)
    env = {}
    if kind == "INVERTED" and fts_version is not None:
        env["LANCE_FTS_FORMAT_VERSION"] = str(fts_version)
    base = Path(base_path)
    failures = []
    # Each setup's aged dataset is built once under from_ref and snapshotted; every
    # exercise for that setup runs on a *copy* of it (a dir copy is far cheaper
    # than rebuilding the index). Cached per shard, keyed by the setup ops.
    snapshots = {}  # tuple(setup) -> (snapshot_path, next_idx), or None if setup failed
    try:
        cases = (
            generate_vector(max_length) if kind == VECTOR_KIND else generate(max_length)
        )
        for i, (setup_tail, exercise) in enumerate(cases):
            if i % num_shards != shard:
                continue
            key = tuple(setup_tail)
            if key not in snapshots:
                snap = base / f"snap_{kind}_{len(snapshots)}"
                shutil.rmtree(snap, ignore_errors=True)
                builder = IndexScenario(kind, snap, setup_tail, [], fts_version)
                try:
                    next_idx = from_venv.execute_method(builder, "setup", env)
                    snapshots[key] = (snap, next_idx)
                except Exception as e:
                    label = describe(
                        kind, from_ref, to_ref, setup_tail, [], fts_version
                    )
                    err = str(e).strip()
                    failures.append({"run": i, "sequence": label, "error": err})
                    snapshots[key] = None
                    shutil.rmtree(snap, ignore_errors=True)
                    if stop_on_first:
                        break
            entry = snapshots[key]
            if entry is None:
                continue  # setup failed; skip its exercises
            snap, next_idx = entry
            ex_path = base / f"ex_{kind}_{i}"
            shutil.rmtree(ex_path, ignore_errors=True)
            shutil.copytree(snap, ex_path)
            scenario = IndexScenario(kind, ex_path, setup_tail, exercise, fts_version)
            scenario.next_idx = next_idx
            label = describe(kind, from_ref, to_ref, setup_tail, exercise, fts_version)
            try:
                to_venv.execute_method(scenario, "exercise_and_check", env)
            except Exception as e:
                error = str(e).strip()
                failures.append({"run": i, "sequence": label, "error": error})
                if stop_on_first:
                    break
            finally:
                shutil.rmtree(ex_path, ignore_errors=True)
    finally:
        for entry in snapshots.values():
            if entry is not None:
                shutil.rmtree(entry[0], ignore_errors=True)
    return failures
