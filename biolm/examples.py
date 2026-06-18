"""Generate SDK usage examples for BioLM models."""
import json
import logging
from typing import Optional, Dict, List, Any, Union
from synchronicity import Synchronizer

import httpx
from biolm.core.http import BioLMApiClient, HttpClient, CredentialsProvider, DEFAULT_TIMEOUT
from biolm.core.const import get_base_domain

log = logging.getLogger("biolm_util")

_synchronizer = Synchronizer()

if not hasattr(_synchronizer, "sync"):
    if hasattr(_synchronizer, "wrap"):
        _synchronizer.sync = _synchronizer.wrap
    if hasattr(_synchronizer, "create_blocking"):
        _synchronizer.sync = _synchronizer.create_blocking
    else:
        from importlib.metadata import version
        raise ImportError(f"Your version of 'synchronicity' ({version('synchronicity')}) is incompatible.")


class ExampleGenerator:
    """Generate SDK usage examples for BioLM models."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the example generator.
        
        Args:
            api_key: Optional API key for authentication.
            base_url: Optional base URL (defaults to BIOLM_BASE_DOMAIN).
        """
        self.base_url = base_url or get_base_domain()
        self.api_key = api_key
        self._headers = CredentialsProvider.get_auth_headers(api_key)
        self._http_client = HttpClient(
            self.base_url.rstrip("/") + "/",
            self._headers,
            DEFAULT_TIMEOUT
        )
    
    async def fetch_community_models(self) -> List[Dict[str, Any]]:
        """
        Fetch list of available models from community-api-models endpoint.
        
        Returns:
            List of model dictionaries with metadata.
        """
        # Try both possible endpoint locations
        endpoints = [
            f"{self.base_url}/api/ui/community-api-models/",
            f"{self.base_url}/ui/community-api-models/",
        ]
        
        for endpoint in endpoints:
            try:
                # Use httpx directly for this endpoint since it might not follow the standard API pattern
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self._headers,
                    timeout=DEFAULT_TIMEOUT
                ) as client:
                    # Try with /api/ui/ prefix first
                    if "/api/ui/" in endpoint:
                        url = endpoint.replace(self.base_url, "")
                    else:
                        url = endpoint.replace(self.base_url, "")
                    
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        # Handle different response formats
                        if isinstance(data, list):
                            return data
                        elif isinstance(data, dict):
                            # Check for common keys that might contain the list
                            for key in ['models', 'results', 'data', 'items']:
                                if key in data and isinstance(data[key], list):
                                    return data[key]
                            # If it's a dict with model info, wrap it
                            return [data] if data else []
                        return []
            except Exception as e:
                log.debug(f"Failed to fetch from {endpoint}: {e}")
                continue
        
        # If all endpoints failed, return empty list
        log.warning("Could not fetch community models from any endpoint")
        return []
    
    async def fetch_model_details(self, model_slug: str, code_examples: bool = False, exclude_docs_html: bool = True) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a specific model from community-api-models endpoint.
        
        Args:
            model_slug: Model slug (e.g., 'esmfold')
            code_examples: If True, include code examples in the response
            exclude_docs_html: If True, exclude HTML documentation
            
        Returns:
            Model details dictionary or None if not found
        """
        # Try both possible endpoint locations
        endpoints = [
            f"{self.base_url}/api/ui/community-api-models/{model_slug}/",
            f"{self.base_url}/ui/community-api-models/{model_slug}/",
        ]
        
        # Build query parameters
        params = {}
        if code_examples:
            params['code_examples'] = 'true'
        if exclude_docs_html:
            params['exclude_docs_html'] = 'true'
        
        for endpoint in endpoints:
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self._headers,
                    timeout=DEFAULT_TIMEOUT
                ) as client:
                    if "/api/ui/" in endpoint:
                        url = endpoint.replace(self.base_url, "")
                    else:
                        url = endpoint.replace(self.base_url, "")
                    
                    resp = await client.get(url, params=params if params else None)
                    if resp.status_code == 200:
                        return resp.json()
                    elif resp.status_code == 404:
                        log.debug(f"Model {model_slug} not found at {endpoint}")
                        return None
            except Exception as e:
                log.debug(f"Failed to fetch model details from {endpoint}: {e}")
                continue
        
        return None
    
    async def get_model_schema(self, model_name: str, action: str) -> Optional[Dict[str, Any]]:
        """
        Get schema for a model and action.
        
        Args:
            model_name: Name of the model.
            action: Action name (encode, predict, generate, lookup).
            
        Returns:
            Schema dictionary or None if not found.
        """
        client = BioLMApiClient(model_name, api_key=self.api_key, base_url=f"{self.base_url}/api/v3")
        try:
            schema = await client.schema(model_name, action)
            await client.shutdown()
            return schema
        except Exception as e:
            log.debug(f"Failed to get schema for {model_name}/{action}: {e}")
            await client.shutdown()
            return None
    
    def _extract_input_type(self, schema: Dict[str, Any]) -> Optional[str]:
        """
        Extract the primary input type from schema.
        
        Args:
            schema: Model schema dictionary.
            
        Returns:
            Input type (e.g., 'sequence', 'pdb', 'context') or None.
        """
        if not schema:
            return None
        
        properties = schema.get('properties', {})
        items_schema = properties.get('items', {})
        
        # Check items schema for type information
        if isinstance(items_schema, dict):
            # items_schema may be the array schema (with nested 'items') or the object schema
            item_props = items_schema.get('properties', {})
            if not item_props and 'items' in items_schema:
                item_props = items_schema.get('items', {}).get('properties', {})
            for key in ['sequence', 'pdb', 'context', 'dna', 'rna']:
                if key in item_props:
                    return key
            
            # Check for oneOf/anyOf patterns
            for pattern_key in ['oneOf', 'anyOf']:
                if pattern_key in items_schema:
                    for option in items_schema[pattern_key]:
                        if isinstance(option, dict):
                            props = option.get('properties', {})
                            for key in ['sequence', 'pdb', 'context', 'dna', 'rna']:
                                if key in props:
                                    return key
        
        # Default to 'sequence' for protein models
        return 'sequence'
    
    def _get_sample_input(self, input_type: str, action: str) -> str:
        """
        Generate sample input value based on input type.
        
        Args:
            input_type: Type of input (sequence, pdb, context, etc.).
            action: Action name.
            
        Returns:
            Sample input string.
        """
        samples = {
            'sequence': 'MSILVTRPSPAGEEL',
            'pdb': 'ATOM      1  N   MET A   1      20.154  16.967  10.410  1.00 20.00           N',
            'context': 'M',
            'dna': 'ATGCGATCGATCG',
            'rna': 'AUGCAUGC',
        }
        return samples.get(input_type, 'SAMPLE_INPUT')
    
    def _generate_python_example(
        self,
        model_name: str,
        action: str,
        input_type: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate Python code example.
        
        Args:
            model_name: Name of the model.
            action: Action name.
            input_type: Type of input.
            schema: Model schema (optional, used to infer input type).
            
        Returns:
            Python code example as string.
        """
        if not input_type:
            input_type = self._extract_input_type(schema) or 'sequence'
        
        sample_input = self._get_sample_input(input_type, action)
        
        # Generate a single, simple example using Model class (recommended pattern)
        example = f"""from biolm import Model

model = Model("{model_name}")
result = model.{action}(items=[{{"{input_type}": "{sample_input}"}}])"""
        
        # Add parameters if schema suggests them (but keep it simple)
        if schema:
            params_schema = schema.get('properties', {}).get('params', {})
            if params_schema and params_schema.get('properties'):
                param_example = {}
                # Only include the first parameter to keep it simple
                for param_name, param_info in list(params_schema.get('properties', {}).items())[:1]:
                    param_type = param_info.get('type', 'string')
                    if param_type == 'number':
                        param_example[param_name] = 0.7
                    elif param_type == 'integer':
                        param_example[param_name] = 10
                    elif param_type == 'boolean':
                        param_example[param_name] = True
                    else:
                        param_example[param_name] = 'value'
                
                if param_example:
                    params_str = json.dumps(param_example)
                    example = f"""from biolm import Model

model = Model("{model_name}")
result = model.{action}(items=[{{"{input_type}": "{sample_input}"}}], params={params_str})"""
        
        return example
    
    def _generate_markdown_example(
        self,
        model_name: str,
        action: str,
        input_type: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate Markdown formatted example."""
        python_code = self._generate_python_example(model_name, action, input_type, schema)
        
        return f"""## {model_name} - {action}

```python
{python_code}
```"""
    
    def _generate_rst_example(
        self,
        model_name: str,
        action: str,
        input_type: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate RST formatted example."""
        python_code = self._generate_python_example(model_name, action, input_type, schema)
        
        # Escape RST special characters in code
        lines = python_code.split('\n')
        indented_code = '\n'.join(f"   {line}" if line else "" for line in lines)
        
        return f"""{model_name} - {action}
{'=' * (len(model_name) + len(action) + 3)}

.. code-block:: python

{indented_code}"""
    
    def _generate_json_example(
        self,
        model_name: str,
        action: str,
        input_type: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate JSON structured example."""
        python_code = self._generate_python_example(model_name, action, input_type, schema)
        
        if not input_type:
            input_type = self._extract_input_type(schema) or 'sequence'
        
        examples_data = {
            "model": model_name,
            "action": action,
            "input_type": input_type,
            "code": python_code
        }
        
        return json.dumps(examples_data, indent=2)
    
    async def generate_example(
        self,
        model_name: str,
        action: Optional[str] = None,
        format: str = 'python',
        input_type: Optional[str] = None
    ) -> str:
        """
        Generate example for a model and action.
        
        Args:
            model_name: Name of the model.
            action: Action name (encode, predict, generate, lookup). If None, generates for all available actions.
            format: Output format ('python', 'markdown', 'rst', 'json').
            input_type: Optional input type override.
            
        Returns:
            Formatted example string.
        """
        # If no action specified, try to get available actions from community models
        if action is None:
            models = await self.fetch_community_models()
            # Try to find model by slug or name (handle both formats)
            model_info = next((
                m for m in models 
                if (m.get('model_slug') == model_name or m.get('slug') == model_name or
                    m.get('model_name') == model_name or m.get('name') == model_name)
            ), None)
            if model_info:
                # Extract actions from boolean flags or actions array
                actions = []
                if 'actions' in model_info and isinstance(model_info['actions'], list):
                    actions = model_info['actions']
                else:
                    # Build actions list from boolean flags
                    if model_info.get('encoder'):
                        actions.append('encode')
                    if model_info.get('predictor'):
                        actions.append('predict')
                    if model_info.get('generator'):
                        actions.append('generate')
                    if model_info.get('classifier'):
                        actions.append('classify')
                    if model_info.get('similarity'):
                        actions.append('similarity')
                
                if actions:
                    # Generate examples for all actions that have schemas
                    all_examples = []
                    for act in actions:
                        schema = await self.get_model_schema(model_name, act)
                        if schema:
                            example = await self._generate_single_example(
                                model_name, act, format, input_type, schema
                            )
                            all_examples.append(example)
                    
                    if all_examples:
                        return "\n\n" + "="*80 + "\n\n".join(all_examples)
                    # If no schemas found, fall through to default actions
                    actions = ['encode', 'predict', 'generate']
                else:
                    # Default to common actions
                    actions = ['encode', 'predict', 'generate']
            else:
                # Default to common actions if model not found
                actions = ['encode', 'predict', 'generate']
            
            # Try each action and use the first one that works
            for act in actions:
                schema = await self.get_model_schema(model_name, act)
                if schema:
                    return await self._generate_single_example(
                        model_name, act, format, input_type, schema
                    )
            
            # Fallback: use first action even without schema
            action = actions[0] if actions else 'encode'
        
        schema = await self.get_model_schema(model_name, action)
        return await self._generate_single_example(model_name, action, format, input_type, schema)
    
    async def _generate_single_example(
        self,
        model_name: str,
        action: str,
        format: str,
        input_type: Optional[str],
        schema: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a single example in the specified format."""
        format_methods = {
            'python': self._generate_python_example,
            'markdown': self._generate_markdown_example,
            'rst': self._generate_rst_example,
            'json': self._generate_json_example,
        }
        
        method = format_methods.get(format, format_methods['python'])
        return method(model_name, action, input_type, schema)
    
    async def shutdown(self):
        """Close HTTP client connections."""
        await self._http_client.close()


# Synchronous wrapper for compatibility
@_synchronizer.sync
class ExampleGeneratorSync(ExampleGenerator):
    pass


# Convenience functions
async def get_example_async(
    model_name: str,
    action: Optional[str] = None,
    format: str = 'python',
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> str:
    """
    Generate SDK usage example for a model (async).
    
    Args:
        model_name: Name of the model.
        action: Action name (optional, will try to detect).
        format: Output format ('python', 'markdown', 'rst', 'json').
        api_key: Optional API key.
        base_url: Optional base URL.
        
    Returns:
        Formatted example string.
    """
    generator = ExampleGenerator(api_key=api_key, base_url=base_url)
    try:
        result = await generator.generate_example(model_name, action, format)
        return result
    finally:
        await generator.shutdown()


def get_example(
    model_name: str,
    action: Optional[str] = None,
    format: str = 'python',
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> str:
    """
    Generate SDK usage example for a model (synchronous).
    
    Args:
        model_name: Name of the model.
        action: Action name (optional, will try to detect).
        format: Output format ('python', 'markdown', 'rst', 'json').
        api_key: Optional API key.
        base_url: Optional base URL.
        
    Returns:
        Formatted example string.
    """
    generator = ExampleGeneratorSync(api_key=api_key, base_url=base_url)
    try:
        return generator.generate_example(model_name, action, format)
    finally:
        generator.shutdown()


async def list_models_async(api_key: Optional[str] = None, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all available models from community-api-models endpoint (async).
    
    Args:
        api_key: Optional API key.
        base_url: Optional base URL.
        
    Returns:
        List of model dictionaries.
    """
    generator = ExampleGenerator(api_key=api_key, base_url=base_url)
    try:
        return await generator.fetch_community_models()
    finally:
        await generator.shutdown()


def list_models(api_key: Optional[str] = None, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all available models from community-api-models endpoint (synchronous).
    
    Args:
        api_key: Optional API key.
        base_url: Optional base URL.
        
    Returns:
        List of model dictionaries.
    """
    generator = ExampleGeneratorSync(api_key=api_key, base_url=base_url)
    try:
        return generator.fetch_community_models()
    finally:
        generator.shutdown()


def get_model_details(
    model_slug: str,
    code_examples: bool = False,
    exclude_docs_html: bool = True,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch detailed information for a specific model (synchronous).
    
    Args:
        model_slug: Model slug (e.g., 'esmfold').
        code_examples: If True, include code examples in the response.
        exclude_docs_html: If True, exclude HTML documentation.
        api_key: Optional API key.
        base_url: Optional base URL.
        
    Returns:
        Model details dictionary or None if not found.
    """
    generator = ExampleGeneratorSync(api_key=api_key, base_url=base_url)
    try:
        return generator.fetch_model_details(model_slug, code_examples, exclude_docs_html)
    finally:
        generator.shutdown()

