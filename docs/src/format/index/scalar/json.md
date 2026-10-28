# JSON Index

The JSON index indexes a single path within a JSON column. It is a wrapper: it
decodes the value at that path out of each JSONB document, converts the decoded
values into one Arrow array, and hands that array to an ordinary scalar index —
the *target index* — which does all of the storage and searching.

A JSON value at a given path has no fixed type, so the wrapper has to pick one
Arrow type for the whole column before the target index can be built:

| JSONB type tag    | `JsonTargetDataType`                 | Value stored in the target index      |
|-------------------|--------------------------------------|---------------------------------------|
| `Boolean`         | `JSON_TARGET_DATA_TYPE_BOOLEAN`      | The decoded boolean                   |
| `Int64`           | `JSON_TARGET_DATA_TYPE_INT64`        | The decoded integer                   |
| `Float64`         | `JSON_TARGET_DATA_TYPE_FLOAT64`      | The decoded float                     |
| `String`          | `JSON_TARGET_DATA_TYPE_UTF8`         | The decoded string, unquoted          |

A JSON null, or a document whose path is absent, is indexed as a null.

The chosen type is recorded in the index details as `target_data_type`. It is
`JSON_TARGET_DATA_TYPE_UNSPECIFIED` for an index written before the details
carried the type, and for one whose type a later rewrite of the details could
not recover.

## Index Details

```protobuf
%%% proto.message.JsonIndexDetails %%%
```

The target index is identified by the `type_url` of `target_details`, which is
read to select the target index implementation before `target_details` itself is
decoded as that implementation's own details message.

## Storage Layout

The JSON index writes no files of its own. Its files are exactly the files of
its target index, written with that index's names and schemas into the same
index directory, and the target's format documentation describes them: for
example [BTree](btree.md) or [Bitmap](bitmap.md).

Reader navigation therefore has two steps. Decode `JsonIndexDetails` to recover
`path` and `target_details`, select the target implementation from the
`type_url` of `target_details`, then navigate the target index exactly as a
standalone index of that type, using `target_details` as its details.

## Accelerated Queries

Only the typed accessor functions are routed to a JSON index, and only when
their path argument is a literal equal to the indexed `path`:

| Function          | Value type evaluated as |
|-------------------|-------------------------|
| `json_get_bool`   | `BOOLEAN`               |
| `json_get_int`    | `INT64`                 |
| `json_get_float`  | `FLOAT64`               |
| `json_get_string` | `UTF8`                  |

Once a predicate is routed, the query types the index can accelerate, and
whether the answer is exact, are those of the target index.

`json_extract` is deliberately not routed. It evaluates to serialized JSON text
while the target index holds decoded native values, so an indexed
`json_extract` predicate would answer a different question than an unindexed
one; quoting is also not order-preserving, so even a `UTF8` target cannot serve
its ranges.

For example, given the documents `{"val": "ab"}` and `{"val": "ab!"}`, a JSON
index on `val` stores the `UTF8` keys `ab` and `ab!`, while
`json_extract(doc, 'val')` evaluates to the text `"ab"` and `"ab!"`, quotes
included:

- `json_extract(doc, 'val') = '"ab"'` matches the first document, but the target
  index holds no key `"ab"`, so an indexed lookup would match nothing.
- `json_extract(doc, 'val') > '"ab"'` does not match the second document, because
  `!` sorts before `"`. In the target index `ab!` sorts after `ab`, so an indexed
  range would match it.
