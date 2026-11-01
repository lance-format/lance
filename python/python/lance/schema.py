# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
from typing import Any, Dict

import pyarrow as pa

from .lance import LanceSchema as LanceSchema
from .lance import _json_to_schema, _schema_to_json


def schema_to_json(schema: pa.Schema) -> Dict[str, Any]:
    """
    Converts a pyarrow schema to a JSON-compatible dict.

    Parameters
    ----------
    schema: pa.Schema
        The PyArrow schema to convert.
    """
    return json.loads(_schema_to_json(schema))


def json_to_schema(schema_json: Dict[str, Any]) -> pa.Schema:
    """
    Converts a JSON-compatible dict to a PyArrow schema.

    Parameters
    ----------
    schema_json: Dict[str, Any]
        The JSON payload to convert to a PyArrow Schema.
    """
    return _json_to_schema(json.dumps(schema_json))
