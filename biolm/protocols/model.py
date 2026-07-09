"""Protocol model — load, validate, execute, and inspect protocol YAML."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from biolm.protocols.validation import (
    ProtocolValidationResult,
    ValidationError,
    load_yaml,
    validate_protocol_file,
)

__all__ = ["Protocol", "ValidationError", "ProtocolValidationResult"]


class Protocol:
    """Load and validate BioLM protocol YAML files.

    Args:
        yaml_path: Path to a protocol YAML file. The file is loaded and
            validated on construction; invalid YAML raises ``ValueError``.

    Use :meth:`validate` as a classmethod to validate without instantiating.
    """
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.data = self._load_yaml(yaml_path)
        self._validate()
    
    def _load_yaml(self, yaml_path: str) -> dict:
        """Load YAML file."""
        return load_yaml(yaml_path)
    
    def _validate(self):
        """Validate protocol against JSON schema (legacy method for __init__)."""
        result = self.validate(self.yaml_path)
        if not result.is_valid:
            if result.errors:
                error = result.errors[0]
                raise ValueError(f"Protocol validation failed: {error.message}")
    
    @classmethod
    def validate(cls, yaml_path: str) -> ProtocolValidationResult:
        """Validate a protocol YAML file."""
        return validate_protocol_file(yaml_path)
    
    @staticmethod
    def _get_examples_dir() -> Path:
        """Get path to examples directory."""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "examples"
    
    @staticmethod
    def _list_available_examples() -> List[str]:
        """List all available example protocol files."""
        examples_dir = Protocol._get_examples_dir()
        if not examples_dir.exists():
            return []
        
        examples = []
        for file_path in examples_dir.glob("*.yaml"):
            # Return name without extension
            examples.append(file_path.stem)
        
        return sorted(examples)
    
    @staticmethod
    def _load_example(name: str) -> str:
        """Load example protocol file content.
        
        Args:
            name: Example name (with or without .yaml extension)
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If example file doesn't exist
        """
        # Strip .yaml extension if provided
        if name.endswith('.yaml') or name.endswith('.yml'):
            name = name.rsplit('.', 1)[0]
        
        examples_dir = Protocol._get_examples_dir()
        example_path = examples_dir / f"{name}.yaml"
        
        if not example_path.exists():
            available = Protocol._list_available_examples()
            raise FileNotFoundError(
                f"Example '{name}' not found. Available examples: {', '.join(available)}"
            )
        
        with open(example_path, 'r') as f:
            return f.read()
    
    @staticmethod
    def _generate_minimal_template(name: Optional[str] = None) -> str:
        """Generate minimal valid protocol YAML template.
        
        Args:
            name: Protocol name (defaults to "My_Protocol")
            
        Returns:
            YAML template as string
        """
        if name is None:
            name = "My_Protocol"
        
        template = f"""name: {name}
schema_version: 1

inputs:
  # Add your input parameters here
  # Example: input_param: string

tasks:
  # Add your tasks here
  # Example:
  # - id: my_task
  #   slug: model-slug
  #   action: predict
  #   request_body:
  #     items: []
"""
        return template
    
    @classmethod
    def init(cls, output_path: str, example: Optional[str] = None, force: bool = False) -> str:
        """Initialize a new protocol YAML file.
        
        Args:
            output_path: Path where the protocol file should be created
            example: Optional example template name to use
            force: If True, overwrite existing file
            
        Returns:
            Path to the created file
            
        Raises:
            FileExistsError: If file exists and force=False
            ValueError: If example name is invalid
            FileNotFoundError: If example file doesn't exist
        """
        output_path_obj = Path(output_path)
        
        # Check if file exists
        if output_path_obj.exists() and not force:
            raise FileExistsError(
                f"File '{output_path}' already exists. Use --force to overwrite."
            )
        
        # Generate protocol content
        if example:
            # Load from example
            content = cls._load_example(example)
        else:
            # Generate minimal template
            # Derive name from filename if not provided
            protocol_name = output_path_obj.stem.replace('_', ' ').title().replace(' ', '_')
            if not protocol_name or protocol_name == '.':
                protocol_name = "My_Protocol"
            content = cls._generate_minimal_template(protocol_name)
        
        # Write file
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, 'w') as f:
            f.write(content)
        
        return str(output_path_obj)
    
    def execute(self, inputs: Optional[Dict[str, Any]] = None):
        """Execute protocol locally with given inputs.

        Args:
            inputs: Input values for the protocol (optional, uses defaults from protocol if not provided).

        Returns:
            :class:`biolm.protocols.runtime.LocalRunResult` with dataframe, records, and metadata.

        Requires:
            ``biolm[pipeline]`` optional dependencies.
        """
        from biolm.protocols.runtime import run_local_protocol

        return run_local_protocol(self.data, inputs=inputs or {})
    
    @classmethod
    async def fetch_by_id_async(cls, protocol_id: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> dict:
        """Fetch a protocol from the platform by ID.
        
        Args:
            protocol_id: Protocol ID to fetch
            api_key: Optional API key for authentication
            base_url: Optional base URL (defaults to BIOLMAI_BASE_API_URL)
            
        Returns:
            Protocol data as dict (same structure as YAML)
            
        Raises:
            FileNotFoundError: If protocol not found (404)
            PermissionError: If not authenticated (401)
            ValueError: If API request fails
        """
        from biolm.core.http import HttpClient, CredentialsProvider
        from biolm.core.const import BIOLM_BASE_API_URL
        import httpx
        
        final_base_url = base_url if base_url is not None else BIOLM_BASE_API_URL
        headers = CredentialsProvider.get_auth_headers(api_key)
        timeout = httpx.Timeout(30.0, connect=10.0)
        
        http_client = HttpClient(final_base_url, headers, timeout)
        
        try:
            endpoint = f"protocols/{protocol_id}/"
            resp = await http_client.get(endpoint)
            
            if resp.status_code == 404:
                await http_client.close()
                raise FileNotFoundError(f"Protocol '{protocol_id}' not found")
            elif resp.status_code == 401:
                await http_client.close()
                raise PermissionError(
                    "Authentication required. Please run 'biolm login' or set BIOLM_TOKEN"
                )
            elif resp.status_code >= 400:
                error_text = resp.text
                try:
                    error_json = resp.json()
                    error_text = error_json.get('error', error_json.get('message', error_text))
                except:
                    pass
                await http_client.close()
                raise ValueError(f"Failed to fetch protocol: {error_text} (status {resp.status_code})")
            
            # Parse JSON response
            try:
                protocol_data = resp.json()
            except Exception as e:
                await http_client.close()
                raise ValueError(f"Invalid JSON response from API: {e}")
            
            if not isinstance(protocol_data, dict):
                await http_client.close()
                raise ValueError(f"Protocol data must be a dictionary, got {type(protocol_data).__name__}")
            
            await http_client.close()
            return protocol_data
            
        except FileNotFoundError:
            await http_client.close()
            raise
        except PermissionError:
            await http_client.close()
            raise
        except Exception as e:
            await http_client.close()
            if isinstance(e, (FileNotFoundError, PermissionError, ValueError)):
                raise
            raise ValueError(f"Failed to fetch protocol: {e}")
    
    @classmethod
    def fetch_by_id(cls, protocol_id: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> dict:
        """Fetch a protocol from the platform by ID (synchronous wrapper).
        
        Args:
            protocol_id: Protocol ID to fetch
            api_key: Optional API key for authentication
            base_url: Optional base URL (defaults to BIOLMAI_BASE_API_URL)
            
        Returns:
            Protocol data as dict (same structure as YAML)
            
        Raises:
            FileNotFoundError: If protocol not found (404)
            PermissionError: If not authenticated (401)
            ValueError: If API request fails
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we need to use a different approach
                # For now, create a new event loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, cls.fetch_by_id_async(protocol_id, api_key, base_url))
                    return future.result()
            else:
                return loop.run_until_complete(cls.fetch_by_id_async(protocol_id, api_key, base_url))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(cls.fetch_by_id_async(protocol_id, api_key, base_url))
    
    @staticmethod
    def render_report(protocol_data: dict, source: str = "file", console=None) -> None:
        """Render a formatted report of the protocol using Rich.
        
        Args:
            protocol_data: Protocol data dictionary
            source: Source description (e.g., "file", "platform")
            console: Optional Rich Console instance (creates one if not provided)
            
        Raises:
            ValueError: If protocol_data is invalid or missing required fields
        """
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        
        if console is None:
            console = Console()
        
        # Validate protocol data
        if not isinstance(protocol_data, dict):
            raise ValueError("Protocol data must be a dictionary")
        
        if not protocol_data:
            raise ValueError("Protocol data is empty")
        
        # Protocol metadata
        name = protocol_data.get("name", "Unknown")
        schema_version = protocol_data.get("schema_version", "N/A")
        description = protocol_data.get("description", "")
        protocol_version = protocol_data.get("protocol_version", "")
        
        # Header panel
        header_parts = [f"[brand.bright]Name:[/brand.bright] {name}"]
        header_parts.append(f"[brand.bright]Schema Version:[/brand.bright] {schema_version}")
        if protocol_version:
            header_parts.append(f"[brand.bright]Protocol Version:[/brand.bright] {protocol_version}")
        if description:
            header_parts.append(f"\n[text]{description}[/text]")
        header_parts.append(f"\n[text.muted]Source: {source}[/text.muted]")
        
        console.print(Panel(
            "\n".join(header_parts),
            title="[bold]Protocol Information[/bold]",
            border_style="brand",
            box=box.ROUNDED,
        ))
        console.print()
        
        # About section
        about = protocol_data.get("about", {})
        if about and isinstance(about, dict):
            about_parts = []
            
            # Title
            if about.get("title"):
                about_parts.append(f"[brand.bright]Title:[/brand.bright] {about['title']}")
            
            # Description
            if about.get("description"):
                about_parts.append(f"\n[text]{about['description']}[/text]")
            
            # Authors
            authors = about.get("authors", [])
            if authors and isinstance(authors, list):
                about_parts.append("\n[brand.bright]Authors:[/brand.bright]")
                for author in authors:
                    if isinstance(author, dict):
                        author_parts = []
                        if author.get("name"):
                            author_parts.append(author["name"])
                        if author.get("affiliation"):
                            author_parts.append(f"({author['affiliation']})")
                        if author.get("email"):
                            author_parts.append(f"<{author['email']}>")
                        if author.get("orcid"):
                            author_parts.append(f"ORCID: {author['orcid']}")
                        if author_parts:
                            about_parts.append("  • " + " ".join(author_parts))
            
            # Keywords
            keywords = about.get("keywords", [])
            if keywords and isinstance(keywords, list):
                keywords_str = ", ".join(str(k) for k in keywords)
                about_parts.append(f"\n[brand.bright]Keywords:[/brand.bright] {keywords_str}")
            
            # DOI
            if about.get("doi"):
                about_parts.append(f"\n[brand.bright]DOI:[/brand.bright] {about['doi']}")
            
            # Links
            links = about.get("links", {})
            if links and isinstance(links, dict):
                about_parts.append("\n[brand.bright]Links:[/brand.bright]")
                for link_name, link_url in links.items():
                    about_parts.append(f"  • {link_name}: [link]{link_url}[/link]")
            
            # Cite (BibTeX)
            if about.get("cite"):
                about_parts.append(f"\n[brand.bright]Citation:[/brand.bright]")
                # Format BibTeX citation in a code block style
                cite_lines = about['cite'].strip().split('\n')
                for line in cite_lines:
                    about_parts.append(f"  [text.muted]{line}[/text.muted]")
            
            if about_parts:
                console.print(Panel(
                    "\n".join(about_parts),
                    title="[bold]About[/bold]",
                    border_style="brand",
                    box=box.ROUNDED,
                ))
                console.print()
        
        # Inputs table
        inputs = protocol_data.get("inputs", {})
        if inputs and isinstance(inputs, dict):
            inputs_table = Table(title="Inputs", show_header=True, header_style="brand.bold", box=box.ROUNDED)
            inputs_table.add_column("Name", style="brand.bright")
            inputs_table.add_column("Type", style="text")
            inputs_table.add_column("Default Value", style="text.muted")
            
            for input_name, input_value in inputs.items():
                # Determine type
                if isinstance(input_value, str):
                    if input_value.startswith("${{"):
                        input_type = "expression"
                        default_val = input_value
                    else:
                        input_type = "string"
                        default_val = f'"{input_value}"' if input_value else '""'
                elif isinstance(input_value, (int, float)):
                    input_type = "number"
                    default_val = str(input_value)
                elif isinstance(input_value, bool):
                    input_type = "boolean"
                    default_val = str(input_value)
                elif isinstance(input_value, list):
                    input_type = "array"
                    default_val = str(input_value)[:50] + "..." if len(str(input_value)) > 50 else str(input_value)
                else:
                    input_type = type(input_value).__name__
                    default_val = str(input_value)[:50] + "..." if len(str(input_value)) > 50 else str(input_value)
                
                inputs_table.add_row(input_name, input_type, default_val)
            
            console.print(inputs_table)
            console.print()
        
        # Tasks summary
        tasks = protocol_data.get("tasks", [])
        if tasks and isinstance(tasks, list):
            task_count = len(tasks)
            model_task_count = sum(
                1 for t in tasks
                if isinstance(t, dict) and t.get("type") != "gather" and ("slug" in t or "class" in t)
            )
            gather_task_count = sum(
                1 for t in tasks
                if isinstance(t, dict) and t.get("type") == "gather"
            )
            
            stats_table = Table(title="Task Summary", show_header=True, header_style="brand.bold", box=box.ROUNDED)
            stats_table.add_column("Metric", style="text")
            stats_table.add_column("Value", style="brand.bright")
            stats_table.add_row("Total Tasks", str(task_count))
            stats_table.add_row("Model Tasks", str(model_task_count))
            stats_table.add_row("Gather Tasks", str(gather_task_count))
            
            console.print(stats_table)
            console.print()
            
            # Task details
            tasks_table = Table(title="Task Details", show_header=True, header_style="brand.bold", box=box.ROUNDED)
            tasks_table.add_column("ID", style="brand.bright")
            tasks_table.add_column("Type", style="text")
            tasks_table.add_column("Model/Action", style="text.muted")
            tasks_table.add_column("Dependencies", style="text.muted")
            
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                
                task_id = task.get("id", "N/A")
                task_type = task.get("type", "model")
                
                # Determine model/action info
                model_info = ""
                if task_type == "gather":
                    model_info = "gather"
                    from_task = task.get("from", "")
                    if from_task:
                        model_info += f" from {from_task}"
                else:
                    # Model task
                    slug = task.get("slug", "")
                    action = task.get("action", "")
                    if slug and action:
                        model_info = f"{slug}/{action}"
                    else:
                        class_name = task.get("class", "")
                        app = task.get("app", "")
                        method = task.get("method", "")
                        if class_name and app and method:
                            model_info = f"{class_name}/{app}/{method}"
                        else:
                            model_info = "N/A"
                
                # Dependencies
                depends_on = task.get("depends_on", [])
                if isinstance(depends_on, list) and depends_on:
                    deps_str = ", ".join(depends_on)
                else:
                    deps_str = "none"
                
                tasks_table.add_row(task_id, task_type, model_info, deps_str)
            
            console.print(tasks_table)
            console.print()
        
        # Outputs - bottom-up: one row per logged field
        outputs = protocol_data.get("outputs", [])
        if outputs and isinstance(outputs, list):
            outputs_table = Table(title="Outputs", show_header=True, header_style="brand.bold", box=box.ROUNDED)
            outputs_table.add_column("Name", style="brand.bright", width=25)
            outputs_table.add_column("Type", style="text", width=12)
            outputs_table.add_column("Where", style="text.muted", width=25)
            outputs_table.add_column("Order By", style="text.muted", width=20)
            outputs_table.add_column("Limit", style="text.muted", width=8)
            
            for output in outputs:
                if not isinstance(output, dict):
                    continue
                
                # Get selection criteria for this output rule
                where = output.get("where")
                order_by = output.get("order_by", [])
                if isinstance(order_by, list) and order_by:
                    order_by_str = ", ".join(
                        f"{item.get('field', '?')} ({item.get('order', '?')})" 
                        for item in order_by if isinstance(item, dict)
                    )
                else:
                    order_by_str = ""
                limit = output.get("limit", 200)
                
                # Format selection criteria for display
                if where:
                    # Strip ${{ and }} from template expressions
                    where_str = str(where)
                    if where_str.startswith("${{") and where_str.endswith("}}"):
                        where_str = where_str[3:-2].strip()
                    where_display = where_str[:23] + "..." if len(where_str) > 23 else where_str
                else:
                    where_display = ""
                if order_by_str:
                    order_by_display = order_by_str[:18] + "..." if len(order_by_str) > 18 else order_by_str
                else:
                    order_by_display = ""
                
                # Extract all logged fields from this output rule
                log = output.get("log", {})
                if isinstance(log, dict):
                    # Params
                    params = log.get("params", {})
                    if isinstance(params, dict):
                        for param_name in params.keys():
                            outputs_table.add_row(
                                param_name,
                                "param",
                                where_display,
                                order_by_display,
                                str(limit)
                            )
                    
                    # Metrics
                    metrics = log.get("metrics", {})
                    if isinstance(metrics, dict):
                        for metric_name in metrics.keys():
                            outputs_table.add_row(
                                metric_name,
                                "metric",
                                where_display,
                                order_by_display,
                                str(limit)
                            )
                    
                    # Tags
                    tags = log.get("tags", {})
                    if isinstance(tags, dict):
                        for tag_name in tags.keys():
                            outputs_table.add_row(
                                tag_name,
                                "tag",
                                where_display,
                                order_by_display,
                                str(limit)
                            )
                    
                    # Aggregates
                    aggregates = log.get("aggregates", [])
                    if isinstance(aggregates, list):
                        for agg in aggregates:
                            if isinstance(agg, dict):
                                field = agg.get("field", "?")
                                ops = agg.get("ops", [])
                                if isinstance(ops, list):
                                    for op in ops:
                                        agg_name = f"{field} ({op})"
                                        outputs_table.add_row(
                                            agg_name,
                                            "aggregate",
                                            where_display,
                                            order_by_display,
                                            str(limit)
                                        )
                    
                    # Artifacts
                    artifacts = log.get("artifacts", [])
                    if isinstance(artifacts, list):
                        for artifact in artifacts:
                            if isinstance(artifact, dict):
                                artifact_name = artifact.get("name", "unnamed")
                                artifact_type = artifact.get("type", "unknown")
                                outputs_table.add_row(
                                    artifact_name,
                                    f"artifact ({artifact_type})",
                                    where_display,
                                    order_by_display,
                                    str(limit)
                                )
            
            if outputs_table.rows:
                console.print(outputs_table)
                console.print()
        
        # Additional metadata
        execution = protocol_data.get("execution", {})
        if execution:
            exec_info = []
            if "concurrency" in execution:
                exec_info.append(f"Concurrency: {execution['concurrency']}")
            if exec_info:
                console.print(Panel(
                    "\n".join(exec_info),
                    title="[bold]Execution Configuration[/bold]",
                    border_style="text.muted",
                    box=box.ROUNDED,
                ))

