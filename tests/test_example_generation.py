"""Tests for SDK example generation functionality."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from biolm.examples import ExampleGenerator, get_example, list_models
from biolm.models import Model


# Mock data for testing
MOCK_COMMUNITY_MODELS = [
    {
        "name": "ESM2-8M",
        "slug": "esm2-8m",
        "actions": ["encode"],
        "description": "ESM2 8M parameter model"
    },
    {
        "name": "ESMFold",
        "slug": "esmfold",
        "actions": ["predict"],
        "description": "Protein structure prediction"
    },
    {
        "name": "ProGen2-OAS",
        "slug": "progen2-oas",
        "actions": ["generate"],
        "description": "Antibody generation"
    }
]

MOCK_SCHEMA_ENCODE = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string"
                    }
                },
                "required": ["sequence"]
            },
            "maxItems": 8
        },
        "params": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "default": 0.7
                }
            }
        }
    },
    "required": ["items"]
}

MOCK_SCHEMA_PREDICT = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string"
                    }
                },
                "required": ["sequence"]
            },
            "maxItems": 1
        }
    },
    "required": ["items"]
}

MOCK_SCHEMA_GENERATE = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string"
                    }
                },
                "required": ["context"]
            },
            "maxItems": 1
        },
        "params": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "default": 0.7
                },
                "num_samples": {
                    "type": "integer",
                    "default": 1
                }
            }
        }
    },
    "required": ["items"]
}


@pytest.fixture
def mock_community_models_response():
    """Mock response from community-api-models endpoint."""
    return MOCK_COMMUNITY_MODELS


@pytest.fixture
def mock_model_schema_encode():
    """Mock schema response for encode action."""
    return MOCK_SCHEMA_ENCODE


@pytest.fixture
def mock_model_schema_predict():
    """Mock schema response for predict action."""
    return MOCK_SCHEMA_PREDICT


@pytest.fixture
def mock_model_schema_generate():
    """Mock schema response for generate action."""
    return MOCK_SCHEMA_GENERATE


@pytest.mark.asyncio
async def test_fetch_community_models_success(mock_community_models_response):
    """Test successful fetching of community models."""
    with patch('biolmai.examples.httpx.AsyncClient') as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_community_models_response
        
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        generator = ExampleGenerator()
        models = await generator.fetch_community_models()
        
        assert isinstance(models, list)
        assert len(models) == 3
        assert models[0]['slug'] == 'esm2-8m'
        await generator.shutdown()


@pytest.mark.asyncio
async def test_fetch_community_models_empty():
    """Test handling of empty response."""
    with patch('biolmai.examples.httpx.AsyncClient') as mock_client:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.get.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        generator = ExampleGenerator()
        models = await generator.fetch_community_models()
        
        assert isinstance(models, list)
        assert len(models) == 0
        await generator.shutdown()


@pytest.mark.asyncio
async def test_get_model_schema_success(mock_model_schema_encode):
    """Test successful schema retrieval."""
    with patch('biolmai.examples.BioLMApiClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.schema = AsyncMock(return_value=mock_model_schema_encode)
        mock_client.shutdown = AsyncMock()
        mock_client_class.return_value = mock_client
        
        generator = ExampleGenerator()
        schema = await generator.get_model_schema("esm2-8m", "encode")
        
        assert schema is not None
        assert schema['properties']['items']['maxItems'] == 8
        await generator.shutdown()


@pytest.mark.asyncio
async def test_get_model_schema_not_found():
    """Test handling of missing schema."""
    with patch('biolmai.examples.BioLMApiClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.schema = AsyncMock(return_value=None)
        mock_client.shutdown = AsyncMock()
        mock_client_class.return_value = mock_client
        
        generator = ExampleGenerator()
        schema = await generator.get_model_schema("nonexistent", "encode")
        
        assert schema is None
        await generator.shutdown()


def test_extract_input_type_sequence(mock_model_schema_encode):
    """Test extraction of input type from schema."""
    from biolm.examples import ExampleGeneratorSync
    generator = ExampleGeneratorSync()
    input_type = generator._extract_input_type(mock_model_schema_encode)
    assert input_type == 'sequence'
    generator.shutdown()


def test_extract_input_type_context(mock_model_schema_generate):
    """Test extraction of context input type."""
    from biolm.examples import ExampleGeneratorSync
    generator = ExampleGeneratorSync()
    input_type = generator._extract_input_type(mock_model_schema_generate)
    assert input_type == 'context'
    generator.shutdown()


def test_get_sample_input():
    """Test sample input generation."""
    from biolm.examples import ExampleGeneratorSync
    generator = ExampleGeneratorSync()
    
    assert generator._get_sample_input('sequence', 'encode') == 'MSILVTRPSPAGEEL'
    assert generator._get_sample_input('context', 'generate') == 'M'
    assert generator._get_sample_input('pdb', 'predict') == 'ATOM      1  N   MET A   1      20.154  16.967  10.410  1.00 20.00           N'
    assert generator._get_sample_input('unknown', 'action') == 'SAMPLE_INPUT'
    
    generator.shutdown()


def test_generate_python_example(mock_model_schema_encode):
    """Test Python example generation."""
    from biolm.examples import ExampleGeneratorSync
    generator = ExampleGeneratorSync()
    
    example = generator._generate_python_example(
        "esm2-8m",
        "encode",
        input_type="sequence",
        schema=mock_model_schema_encode
    )
    
    assert "from biolm import Model" in example
    assert "Model(\"esm2-8m\")" in example
    assert "model.encode" in example
    # Should be a single, simple example (not multiple variations)
    assert example.count("from biolm import Model") == 1
    
    generator.shutdown()


def test_generate_markdown_example():
    """Test Markdown example generation."""
    from biolm.examples import ExampleGeneratorSync
    generator = ExampleGeneratorSync()
    
    example = generator._generate_markdown_example(
        "esm2-8m",
        "encode",
        input_type="sequence"
    )
    
    assert "## esm2-8m - encode" in example
    assert "```python" in example
    assert "from biolm import Model" in example
    
    generator.shutdown()


def test_generate_rst_example():
    """Test RST example generation."""
    from biolm.examples import ExampleGeneratorSync
    generator = ExampleGeneratorSync()
    
    example = generator._generate_rst_example(
        "esm2-8m",
        "encode",
        input_type="sequence"
    )
    
    assert "esm2-8m - encode" in example
    assert ".. code-block:: python" in example
    assert "from biolm import Model" in example
    
    generator.shutdown()


def test_generate_json_example():
    """Test JSON example generation."""
    from biolm.examples import ExampleGeneratorSync
    generator = ExampleGeneratorSync()
    
    example = generator._generate_json_example(
        "esm2-8m",
        "encode",
        input_type="sequence"
    )
    
    data = json.loads(example)
    assert data['model'] == 'esm2-8m'
    assert data['action'] == 'encode'
    assert 'code' in data
    assert 'from biolm import Model' in data['code']
    
    generator.shutdown()


@pytest.mark.asyncio
async def test_generate_example_with_schema(mock_model_schema_encode):
    """Test example generation with schema."""
    with patch('biolmai.examples.ExampleGenerator.get_model_schema') as mock_schema:
        mock_schema.return_value = mock_model_schema_encode
        
        generator = ExampleGenerator()
        example = await generator.generate_example("esm2-8m", "encode", "python")
        
        assert "from biolm import Model" in example
        assert "esm2-8m" in example
        assert "encode" in example
        
        await generator.shutdown()


@pytest.mark.asyncio
async def test_generate_example_without_schema():
    """Test example generation without schema (fallback)."""
    with patch('biolmai.examples.ExampleGenerator.get_model_schema') as mock_schema:
        mock_schema.return_value = None
        
        generator = ExampleGenerator()
        example = await generator.generate_example("esm2-8m", "encode", "python")
        
        # Should still generate example with defaults
        assert "from biolm import Model" in example
        assert "esm2-8m" in example
        
        await generator.shutdown()


def test_get_example_function():
    """Test standalone get_example function."""
    with patch('biolmai.examples.ExampleGeneratorSync') as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_example.return_value = "example code"
        mock_generator.shutdown = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        result = get_example("esm2-8m", "encode", "python")
        
        assert result == "example code"
        mock_generator.generate_example.assert_called_once_with("esm2-8m", "encode", "python")
        mock_generator.shutdown.assert_called_once()


def test_list_models_function():
    """Test standalone list_models function."""
    with patch('biolmai.examples.ExampleGeneratorSync') as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.fetch_community_models.return_value = MOCK_COMMUNITY_MODELS
        mock_generator.shutdown = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        result = list_models()
        
        assert isinstance(result, list)
        assert len(result) == 3
        mock_generator.fetch_community_models.assert_called_once()
        mock_generator.shutdown.assert_called_once()


def test_model_get_example():
    """Test Model.get_example() method."""
    with patch('biolmai.models.ExampleGeneratorSync') as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_example.return_value = "example code"
        mock_generator.shutdown = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        model = Model("esm2-8m")
        result = model.get_example("encode", "python")
        
        assert result == "example code"
        mock_generator.generate_example.assert_called_once_with("esm2-8m", "encode", "python")
        mock_generator.shutdown.assert_called_once()


def test_model_get_examples():
    """Test Model.get_examples() method."""
    with patch('biolmai.models.ExampleGeneratorSync') as mock_generator_class:
        mock_generator = MagicMock()
        mock_generator.generate_example.return_value = "all examples"
        mock_generator.shutdown = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        model = Model("esm2-8m")
        result = model.get_examples("python")
        
        assert result == "all examples"
        mock_generator.generate_example.assert_called_once_with("esm2-8m", None, "python")
        mock_generator.shutdown.assert_called_once()


def test_example_with_parameters(mock_model_schema_generate):
    """Test example generation includes parameters when schema has them."""
    from biolm.examples import ExampleGeneratorSync
    generator = ExampleGeneratorSync()
    
    example = generator._generate_python_example(
        "progen2-oas",
        "generate",
        input_type="context",
        schema=mock_model_schema_generate
    )
    
    # Should include params example
    assert "params" in example.lower() or "temperature" in example
    
    generator.shutdown()


def test_example_formats(mock_model_schema_encode):
    """Test all output formats are valid."""
    from biolm.examples import ExampleGeneratorSync
    generator = ExampleGeneratorSync()
    
    # Test format methods directly (they're not async)
    python_example = generator._generate_python_example(
        "esm2-8m", "encode", "sequence", mock_model_schema_encode
    )
    assert isinstance(python_example, str)
    assert len(python_example) > 0
    
    markdown_example = generator._generate_markdown_example(
        "esm2-8m", "encode", "sequence"
    )
    assert isinstance(markdown_example, str)
    assert len(markdown_example) > 0
    
    rst_example = generator._generate_rst_example(
        "esm2-8m", "encode", "sequence"
    )
    assert isinstance(rst_example, str)
    assert len(rst_example) > 0
    
    json_example = generator._generate_json_example(
        "esm2-8m", "encode", "sequence"
    )
    assert isinstance(json_example, str)
    # Validate JSON format
    data = json.loads(json_example)
    assert isinstance(data, dict)
    
    generator.shutdown()


def test_example_error_handling():
    """Test error handling in example generation."""
    from biolm.examples import ExampleGeneratorSync
    generator = ExampleGeneratorSync()
    
    # Should handle None schema gracefully
    example = generator._generate_python_example(
        "esm2-8m",
        "encode",
        input_type="sequence",
        schema=None
    )
    
    assert "from biolm import Model" in example
    assert len(example) > 0
    
    generator.shutdown()

