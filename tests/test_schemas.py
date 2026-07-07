import pytest

from biolm.core.http import BioLMApiClient


@pytest.mark.asyncio
async def test_schema_max_items_is_int_live():
    # Use a real model and action that exist on the API
    model_name = "esm2-35m"  # or any model you know exists
    action = "encode"      # or "encode", "generate", etc.

    client = BioLMApiClient(model_name)
    schema = await client.schema(model_name, action)
    assert schema is not None, "Schema should not be None"
    max_items = client.extract_max_items(schema)
    assert isinstance(max_items, int), f"maxItems should be int, got {type(max_items)}: {max_items}"
    # Optionally, print or log the value for debugging
    print(f"maxItems for {model_name}/{action}: {max_items}")

@pytest.mark.asyncio
async def test_schema_contains_throttle_rate_live():
    model_name = "esm2-35m"  # Use a real model name available on your API
    action = "encode"        # Use a real action, e.g., "encode", "generate", etc.

    client = BioLMApiClient(model_name)
    schema = await client.schema(model_name, action)
    assert schema is not None, "Schema should not be None"
    assert "throttle_rate" in schema, f"'throttle_rate' key not found in schema: {schema.keys()}"
    # Optionally, print the value for debugging
    print(f"throttle_rate for {model_name}/{action}: {schema['throttle_rate']}")
