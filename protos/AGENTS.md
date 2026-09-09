# Protobuf Guidelines

Also see [root AGENTS.md](../AGENTS.md) for cross-language standards.

## Change Process

- Changes to protobuf schemas that define a persisted Lance format require a PMC vote on the pull request, enforced by the `format-spec-vote` CI gate. See [Lance Format Specification Changes](../docs/src/community/voting.md#lance-format-specification-changes).
- Keep a persisted-format proto change in its own PR, together with the matching `docs/src/format/` change and only the library edits needed to compile. Put the implementation in a follow-up PR — voters need to read the contract, not its implementation.
- Execution-plan schemas (`ann.proto`, `filtered_read.proto`, and `table_identifier.proto`) are wire contracts, not persisted Lance formats. Changes to them belong with their implementation and do not require a format vote or a `docs/src/format/` change.

## Compatibility

- Protobuf schemas that are part of a stable file format or any other stable persisted contract must remain backwards compatible. Never reuse or change their existing field numbers.
- Protobuf schemas used exclusively by an unstable file format follow the root file-format stability contract: do not preserve compatibility with prior unstable revisions. Before making a breaking protobuf change, verify that the schema is not shared with a stable format or another persisted contract.
- Execution wire schemas may still cross process or version boundaries. Preserve their field-number compatibility unless all producers and consumers are upgraded atomically.

## Schema Design

- Use `optional` when you need to distinguish "not set" from "zero value" — `optional` enables presence tracking (`has_*` methods) and maps to `Option<T>` in Rust. Bare proto3 fields have no presence semantics: they always hold a value (defaulting to zero), so you cannot tell if the sender explicitly set them.
- Use structured message types (e.g., `BasePath`) instead of plain scalars, and scope fields to operation-specific messages (e.g., `InsertTransaction`) rather than generic top-level ones.
- Don't duplicate data across messages — store each fact once and derive relationships. Prefer parallel sequences over maps when keys already exist in another field.

## Documentation

- Document the semantic meaning of both present and absent states for `optional` fields — explain when each case applies.
- Use precise domain terminology in field descriptions — avoid ambiguous abbreviations or terms that collide with domain concepts.
- Write multi-line comments as `/* text` / ` * text` / ` */` blocks so editors can fold them; keep single-line comments as `//`. The `ci/check_proto_comments.py` check (pre-commit hook and the `Protobuf lint` workflow) enforces this and can rewrite offenders with `--fix`.
