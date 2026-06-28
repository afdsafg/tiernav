"""Schema export gate for public runtime contracts."""

import json

from src.tiernav_runtime.contracts import PUBLIC_MODELS, dump_runtime_json_schemas


def test_schemas_are_json_serializable_sorted():
    schemas = dump_runtime_json_schemas()
    encoded = json.dumps(schemas, sort_keys=True)
    assert "RunSpec" in encoded
    assert "EpisodeResult" in encoded


def test_all_public_model_schemas_present():
    schemas = dump_runtime_json_schemas()
    assert set(schemas.keys()) == set(PUBLIC_MODELS.keys())
    for name, schema in schemas.items():
        assert isinstance(schema, dict), f"{name} schema is not a dict"
        assert "properties" in schema, f"{name} missing properties"
