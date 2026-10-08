# Tokenizers

Currently, Lance has built-in support for ICU, Jieba, and Lindera. ICU uses built-in segmenter data. Jieba and Lindera require external language models.
If tokenization is needed, you can download language models by yourself.
You can specify the location where the language models are stored by setting the environment variable LANCE_LANGUAGE_MODEL_HOME.
If it's not set, the default value is

```bash
${system data directory}/lance/language_models
```

It also supports configuring user dictionaries,
which makes it convenient for users to expand their own dictionaries without retraining the language models.

## Inspect Query Tokenization

Use `lance.tokenize` to inspect the tokens that a full-text query will produce
without creating a dataset or index:

```python
import lance

tokens = lance.tokenize("the Cats and Dogs")
[(token.text, token.position) for token in tokens]
# [("cat", 0), ("dog", 2)]
```

Positions start at the first retained query token and preserve gaps left by stop
word removal and other filters. This is the same representation used for phrase
matching. The function accepts the tokenizer-related options supported by
`LanceDataset.create_scalar_index`, including custom stop words, n-grams, and the
code analyzer:

```python
tokens = lance.tokenize(
    "getUserName::value42",
    analyzer="code",
    split_identifiers=True,
    index_operators=True,
)
```

Options set to `None` use the selected analyzer profile's default. For example,
the code analyzer disables stemming and stop-word removal unless explicitly
overridden. The exception is `max_token_length`: omitting it keeps the default
length limit of 40, while `max_token_length=None` disables the limit.

## ICU Tokenizer

ICU uses Unicode word boundary rules and bundled dictionary data for complex scripts. It is useful for mixed-language text and does not require downloading a language model.

```python
ds.create_scalar_index("text", "INVERTED", base_tokenizer="icu")
```

Use `icu/split` when mixed-language text also contains punctuation-delimited identifiers that should be searchable by part.

```python
ds.create_scalar_index("text", "INVERTED", base_tokenizer="icu/split")
```

## Code Analyzer

The code analyzer tokenizes code-like text so identifiers and operators can be
searched. Enable it with `analyzer="code"`; `base_tokenizer="code"` is
equivalent and infers the same profile:

```python
ds.create_scalar_index("source", "INVERTED", analyzer="code")
```

Indexes built with the code analyzer require FTS format v3.

Identifiers are Unicode alphanumeric characters plus `_`. Everything else is a
lexical boundary, and punctuation that is not an indexed operator (for example
`.` `@` `#` `$`) is dropped.

The code profile changes the defaults and adds four flags:

| Option | Default | Description |
| --- | --- | --- |
| `split_identifiers` | `False` | Split identifiers such as `getUserName` into subwords (`get`, `User`, `Name`). |
| `split_on_numerics` | `True` | Split subwords at letter/number boundaries (for example `value42` into `value` and `42`). |
| `preserve_original` | `True` | Index the complete identifier in addition to its subwords. |
| `index_operators` | `False` | Index operators such as `::`, `->`, and `!=`. |

`split_on_numerics` and `preserve_original` only take effect when
`split_identifiers=True`. The code profile also disables stemming and stop-word
removal, and keeps the default `max_token_length` of 40.

Because `split_identifiers` defaults to `False`, `getUserName` is not found by
`user`; enable splitting to match subwords:

```python
tokens = lance.tokenize("getUserName::value42", analyzer="code")
[(token.text, token.position) for token in tokens]
# [("getusername", 0), ("value42", 1)]

tokens = lance.tokenize(
    "getUserName::value42",
    analyzer="code",
    split_identifiers=True,
    index_operators=True,
)
[(token.text, token.position) for token in tokens]
# [("getusername", 0), ("get", 0), ("user", 1), ("name", 2),
#  ("::", 3), ("value42", 4), ("value", 4), ("42", 5)]
```

When identifiers are split, subwords are assigned consecutive positions starting
at the identifier's position, and the preserved original identifier is stored at
the first position with a length covering all subwords. With
`with_position=True`, phrase queries can match subword sequences inside a single
identifier, such as `'user name'` within `getUserName`.

## Language Models of Jieba

### Downloading the Model

```bash
python -m lance.download jieba
```

The language model is stored by default in `${LANCE_LANGUAGE_MODEL_HOME}/jieba/default`.

### Using the Model

```python
ds.create_scalar_index("text", "INVERTED", base_tokenizer="jieba/default")
```

### User Dictionaries

Create a file named config.json in the root directory of the current model.

```json
{
    "main": "dict.txt",
    "users": ["path/to/user/dict.txt"]
}
```

- The "main" field is optional. If not filled, the default is "dict.txt".
- "users" is the path of the user dictionary. For the format of the user dictionary, please refer to https://github.com/messense/jieba-rs/blob/main/jieba/src/data/dict.txt.

## Language Models of Lindera

### Downloading the Model

```bash
python -m lance.download lindera -l [ipadic|ko-dic|unidic]
```

Note that the language models of Lindera need to be compiled. Please install lindera-cli first. For detailed steps, please refer to https://github.com/lindera/lindera/tree/main/lindera-cli.

The language model is stored by default in ${LANCE_LANGUAGE_MODEL_HOME}/lindera/[ipadic|ko-dic|unidic]

### Using the Model

```python
ds.create_scalar_index("text", "INVERTED", base_tokenizer="lindera/ipadic")
```

### User Dictionaries

Create a file named config.yml in the root directory of your model, or specify a custom YAML file using the `LINDERA_CONFIG_PATH` environment variable.
If both are provided, the config.yml in the root directory will be used.
For more detailed configuration methods, see the lindera documentation at https://github.com/lindera/lindera/.

```yaml
segmenter:
    mode: "normal"
    dictionary: /path/to/lindera/ipadic/main
```

## Create your own language model

Put your language model into `LANCE_LANGUAGE_MODEL_HOME`.
