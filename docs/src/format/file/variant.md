# Variant in Lance 2.3

This representation is reserved and not yet implemented. Defining the descriptor
does not enable the Variant capability or authorize writers to emit it.

This contract applies only to the unstable 2.3 file format. It separates a
Variant's logical identity from its file-local shredding layout. The binary
values and metadata use the [Parquet Variant encoding](https://github.com/apache/parquet-format/blob/master/VariantEncoding.md);
the recursive layout follows the [Parquet Variant shredding specification](https://github.com/apache/parquet-format/blob/master/VariantShredding.md).
Lance column pages use the existing 2.3 structural and value encodings.

## Logical schema and compatibility

A logical Variant is a `lance.file.Field` with type `LEAF`, logical type
`variant`, one nonnegative field ID, and no children. Its metadata includes
`ARROW:extension:name = arrow.parquet.variant`. Shredding choices, physical
children, and local field IDs do not appear in the table schema.

The Arrow representation returned by a reader is always unshredded. For a
nullable field named `event`, its storage schema is:

```python
pa.schema([
    pa.field("event", pa.struct([
        pa.field("metadata", pa.binary(), nullable=False),
        pa.field("value", pa.binary(), nullable=False),
    ]), nullable=True, metadata={
        "ARROW:extension:name": "arrow.parquet.variant",
    }),
])
```

The enclosing struct's validity represents SQL NULL. A non-null struct whose
`value` encodes Variant null represents Variant null. Missing object fields
remain absent from the object; they do not become fields containing null.
Reconstruction preserves values and native Variant types without a conversion
through JSON. Byte-identical metadata dictionaries and integer storage widths
are not required.

Manifest reader and writer feature words must both set `FLAG_VARIANT` (`2^20`)
when their schema contains a logical Variant, including an empty table. Readers
must check this feature before interpreting the schema. Writers that do not
implement this contract must refuse the table. Once activated, both bits remain
set in derived snapshots, including after dropping the logical field or restoring
an older version: retained files may still describe Variant groups. Historical
snapshots retain their original flags and schemas.

Files carrying logical Variants must use version 2.3. Writers must reject a
request to persist a logical Variant using 2.2 or earlier. Existing stable files
containing ordinary structs with Variant extension metadata retain their
existing interpretation, child IDs, and projection behavior. They are not
silently reinterpreted as logical Variant leaves. Converting such a column to
this representation requires an explicit schema conversion and data rewrite.
Existing JSONB columns are unchanged.

## File descriptor and physical column mapping

Global buffer zero of a 2.3 file contains `lance.file23.FileDescriptor`, defined
in `file_v2_3.proto`. Its `schema` is the logical file schema and its `length`
is the top-level row count. `variant_layouts` contains exactly one entry per
logical Variant in that schema, in schema preorder, even when no path is shredded.
An entry identifies its logical field by ID and supplies a local physical schema.
Missing, duplicate, or extraneous entries are invalid.

Physical columns are enumerated by traversing the logical file schema in order.
Ordinary fields retain the 2.3 physical column enumeration rules. At a Variant
leaf, substitute its local physical schema and enumerate that schema's primitive
leaves in preorder. Those leaves occupy one contiguous range of column indices.
The number of columns is derived from the local schema, not duplicated in the
descriptor. The final count must equal the footer's column count.

In a table's `DataFile.fields` and `DataFile.column_indices`, the logical Variant
ID maps once to the first physical column of this range. Local field IDs never
enter these mappings or consume IDs from the table's field-ID allocator.
Columns following a Variant are mapped using the actual local leaf count, which
can differ between files. Readers must load each file's descriptor before
expanding a logical projection or validating its column mapping.

The local schema has exactly one struct root named `variant`. Field IDs are
consecutive preorder integers starting at zero; the root's parent ID is -1 and
all other parent IDs refer to local ancestors. Schema and field metadata are
empty, and key annotations and legacy dictionary descriptors are absent. These local
IDs describe nesting only. A local field name containing punctuation is a
literal object key, not a table-column path.

Variant leaves may occur inside ordinary structs or lists. Their local schema
is substituted at that nesting position, retaining all ancestor validity and
list structure. Local columns therefore use the same inherited structural
domains as the replaced leaf. They must agree on parent validity, list lengths,
and element order; physical pages need not have identical row boundaries.

## Physical schema grammar

The root's nullability equals that of the logical field. Its children, in order,
are required `metadata: binary`, nullable `value: binary`, and an optional
nullable `typed_value`. The last child is absent when there is no shredding.
`metadata` supplies the dictionary for every residual value in that row,
including nested residuals, and includes the names of shredded object fields.
SQL-null parents make child contents inaccessible through the structural plane.

Each nested Variant node is a required struct with nullable `value: binary`
followed by an optional nullable `typed_value`. A typed value is either a
supported primitive, a struct of named nodes, or a list of required nodes.
Object-node names are unique and ordered lexicographically by unsigned UTF-8
bytes. A typed object layout must select at least one field. An object without
selected fields is represented by a node with only `value`.

This revision permits boolean, int8/int16/int32/int64, float32/float64,
binary/large-binary, and string/large-string typed primitives. The corresponding
Lance logical types are `bool`, `int8`, `int16`, `int32`, `int64`, `float`,
`double`, `binary`, `large_binary`, `string`, and `large_string`. Typed arrays
use `list.struct` or `large_list.struct`, with one required element node.
No other physical types are permitted by this revision. Native decimal,
temporal, and UUID values remain representable in binary Variant residuals.
Writers must reject unsupported requested target types before writing a file.
Offsets must be representable by the selected Arrow type; exceeding a
representation limit is an error, never a truncation.

A physical schema shredding `id` as int64 and `name` as string is:

```python
pa.schema([
    pa.field("variant", pa.struct([
        pa.field("metadata", pa.binary(), nullable=False),
        pa.field("value", pa.binary()),
        pa.field("typed_value", pa.struct([
            pa.field("id", pa.struct([
                pa.field("value", pa.binary()),
                pa.field("typed_value", pa.int64()),
            ]), nullable=False),
            pa.field("name", pa.struct([
                pa.field("value", pa.binary()),
                pa.field("typed_value", pa.string()),
            ]), nullable=False),
        ])),
    ])),
])
```

## Writing and reconstruction

Fallback is per node, not per row. A writer first distinguishes SQL-null parents
from present Variant values, then recursively applies these rules:

1. Without a typed layout, store the complete node in `value`.
2. With a primitive layout, store a value in `typed_value` only when its Variant
   kind matches the target and its value is representable. Otherwise store it
   in `value`. Never write both. Signed integer widths may change when the value
   fits; string and binary offset widths may change. Floating-point values
   require a matching-width target and preserve signed zero, infinities, and
   NaN semantics. Do not coerce booleans or strings to numbers, convert integers
   to floating point, or truncate fractional numbers to integers.
3. With an object layout, an object always has a non-null `typed_value`, including
   an empty object or an object missing every selected field. Recurse into each
   selected field. A missing field has both children null; a present Variant
   null is encoded in the field's `value`. The parent's residual contains only
   unselected fields and is null when none remain. It must never contain a
   selected field, even when that field does not fit its target type. A non-object
   instead uses only the parent's `value`.
4. With a list layout, an array always has a non-null typed list and null parent
   `value`. Recurse into each element in order. An empty array has a valid empty
   list. Null elements use Variant null in the element's `value`, not a missing
   element. A non-array uses only the parent's `value`.

For `{"id": 5, "name": false}` under the schema above, the root typed object is
present, `id.typed_value` is 5, and `name.value` encodes false. Neither field is
present in the root residual. A type mismatch in `name` cannot hide `id` from
its physical column.

To reconstruct a node, first check ancestor validity. A null logical root is
SQL NULL. For a present node with a typed object, recursively reconstruct each
selected field, omit missing fields, and union them with the residual object's
unselected fields. Reject overlapping keys or a non-object residual. For a
typed list, reconstruct every element in order and reject a non-null parent
residual. For a typed primitive, convert it to its corresponding Variant value
and reject a non-null residual. If only `value` is present, decode it with the
root metadata. If both are null, the field is missing; in a context requiring a
value, return Variant null as specified by Parquet. Invalid binary encodings,
inconsistent structural domains, or contradictory populated nodes are errors.

## Reads, predicates, and layout evolution

The reference read operation reconstructs the logical value and then extracts
the requested path. Ordinary table projections cannot address physical paths
such as `event.typed_value.id.typed_value`. Physical children are not logical
schema fields or SQL columns.

A path-read optimization may read a subset of physical columns only when it
preserves the reference result. It must account for ancestor validity, object
presence, list positions, missing fields, Variant nulls, and per-node residuals.
If the descriptor does not shred the requested path, read the residual of the
nearest applicable ancestor. If a node does shred it, inspect both its typed and
residual representations where either can contain values relevant to the query.
A reader may omit irrelevant siblings, but may not treat a null typed value as
proof that the logical path is absent.

Statistics pruning is valid only for the rows covered by those statistics and
the query's actual conversion semantics. Typed statistics alone cannot exclude
rows whose residual might satisfy the predicate. A sufficient condition for
using typed comparisons is that the corresponding residual is all null over the
covered rows and the comparison is valid for the typed domain. Ancestor
validity and null predicates must still be honored. This contract does not add
a statistics or index format.

Append may choose a new local layout without changing the table schema or the
logical field ID. A scan reconstructs each file independently and returns the
same unshredded Arrow type across all fragments. A new layout affects only newly
written files. Updates and compaction reconstruct logical values and write new
files with their chosen layouts; they cannot copy an old layout descriptor onto
new physical columns. Layout changes do not allocate logical field IDs, alter
logical index expressions, or change values. Existing transaction and index
maintenance rules still apply to rewritten rows. Time travel uses the original
files and their descriptors; changing layouts does not modify old snapshots.

Automatic path selection, write-API configuration, and query planning policies
are implementation concerns. They must not change this logical or physical
contract. Storing only residuals is a valid layout and a correctness baseline.
