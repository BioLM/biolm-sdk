"""Console script for biolmai."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union, Any, List, Dict
import json
import builtins

import click
from click.formatting import HelpFormatter
from biolm import __version__ as BIOLM_VERSION
from biolm.cli_theme import create_console, no_color_requested
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import box

from biolm.core.auth import (
    are_credentials_valid,
    generate_access_token,
    get_auth_status,
    oauth_login,
    save_access_refresh_token,
)
from biolm.core.const import (
    ACCESS_TOK_PATH,
    BIOLM_BASE_API_URL,
    BIOLM_BASE_DOMAIN,
    get_model_catalog_base,
)
from biolm.examples import get_example, list_models, get_model_details
from biolm.io import load_fasta, load_csv, load_pdb, load_json, to_fasta, to_csv, to_pdb, to_json
from biolm.models import Model
from biolm.datasets_mlflow import (
    MLflowNotAvailableError,
    list_datasets,
    get_dataset,
    upload_dataset,
    download_dataset,
)

console = create_console()

# Common argument descriptions for better help text
ARGUMENT_DESCRIPTIONS = {
    'filename': 'Protocol file name (default: protocol.yaml)',
    'model_name': 'Name of the model',
    'action': 'Action to perform (encode, predict, generate, lookup)',
    'workspace_id': 'Workspace identifier',
    'name': 'Name for the resource',
    'protocol_source': 'Protocol file path or protocol ID',
    'protocol_file': 'Path to protocol YAML file',
    'output_path': 'Output directory path',
    'results': 'Path to results file',
    'dataset_id': 'Dataset identifier',
    'file_path': 'Path to file',
}


class RichHelpFormatter(click.HelpFormatter):
    """Custom help formatter using Rich for styled output."""
    
    def write_usage(self, prog, args='', prefix='Usage: '):
        """Write usage line with Rich formatting."""
        usage_text = f"{prefix}[brand.bright]{prog}[/brand.bright] [OPTIONS] COMMAND [ARGS]..."
        console.print(usage_text)
        console.print()
    
    def write_heading(self, heading):
        """Write section heading with Rich formatting."""
        console.print(f"[bold]{heading}[/bold]")
    
    def write_dl(self, rows, col_max=30, col_spacing=2):
        """Write definition list (command/option + description) with Rich formatting."""
        for primary, secondary in rows:
            # Format primary (command/option name) in brand color
            primary_text = Text(primary, style="brand.bright")
            # Format secondary (description) in default text
            secondary_text = Text(secondary or "", style="text")
            
            # Create a two-column layout
            # Calculate padding for alignment
            padding = max(0, col_max - len(primary))
            console.print(f"  {primary_text}{' ' * padding}  {secondary_text}")
        console.print()


class RichCommand(click.Command):
    """Custom Click Command with Rich help formatting."""
    
    def format_help(self, ctx, formatter):
        """Format help output using Rich with styled organization."""
        # Write usage
        self.write_usage(ctx, formatter)
        
        # Write description
        if self.help:
            # Get first line of help as description
            desc_lines = self.help.split('\n')
            first_line = desc_lines[0].strip()
            if first_line:
                console.print(f"[text]{first_line}[/text]")
                console.print()
            
            # Write additional description lines if present
            if len(desc_lines) > 1:
                # Preserve blank lines to maintain spacing in Examples section
                for line in desc_lines[1:]:
                    stripped = line.strip()
                    if stripped:
                        # Style comments (starting with #) with muted color
                        if stripped.startswith('#'):
                            console.print(f"[text.muted]{stripped}[/text.muted]")
                        # Style command examples (starting with biolm) with brand color
                        elif stripped.startswith('biolm'):
                            console.print(f"[brand.bright]{stripped}[/brand.bright]")
                        else:
                            # Regular text for descriptions
                            console.print(f"[text]{stripped}[/text]")
                    else:
                        # Preserve blank lines for better example spacing
                        console.print()
                console.print()
        
        # Write Arguments section if present (before Options)
        args = []
        for param in self.get_params(ctx):
            if isinstance(param, click.Argument):
                rv = param.get_help_record(ctx)
                if rv:
                    # Has help record from Click
                    args.append(rv)
                else:
                    # No help record, but still show the argument
                    # Get argument name and description from param
                    arg_name = param.name.upper()
                    # Try to get help from param attribute, or use our mapping
                    arg_help = getattr(param, 'help', None) or ARGUMENT_DESCRIPTIONS.get(param.name, '')
                    args.append((arg_name, arg_help))
        if args:
            # Create box content
            box_content = []
            for arg_name, arg_help in args:
                # Format argument name in brand color, description in text
                arg_padding = " " * max(0, 25 - len(arg_name))
                if arg_help:
                    line = f"[brand.bright]{arg_name}[/brand.bright]{arg_padding}  [text]{arg_help}[/text]"
                else:
                    line = f"[brand.bright]{arg_name}[/brand.bright]"
                box_content.append(line)
            
            # Create panel with box style
            panel = Panel(
                "\n".join(box_content),
                title="[bold]Arguments[/bold]",
                border_style="border",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            console.print(panel)
            console.print()
        
        # Write Options section with box
        opts = []
        for param in self.get_params(ctx):
            if isinstance(param, click.Option) and not param.hidden:
                rv = param.get_help_record(ctx)
                if rv:
                    opts.append(rv)
        if opts:
            # Create box content
            box_content = []
            for opt_name, opt_help in opts:
                # Format option name in brand color, description in text
                opt_padding = " " * max(0, 25 - len(opt_name))
                if opt_help:
                    line = f"[brand.bright]{opt_name}[/brand.bright]{opt_padding}  [text]{opt_help}[/text]"
                else:
                    line = f"[brand.bright]{opt_name}[/brand.bright]"
                box_content.append(line)
            
            # Create panel with box style
            panel = Panel(
                "\n".join(box_content),
                title="[bold]Options[/bold]",
                border_style="border",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            console.print(panel)
            console.print()
    
    def write_usage(self, ctx, formatter):
        """Write usage line with Rich formatting."""
        # Build usage string similar to Click's format
        usage_parts = [ctx.command_path]
        
        # Add options marker if there are any options
        has_options = any(isinstance(p, click.Option) and not p.hidden 
                         for p in self.get_params(ctx))
        if has_options:
            usage_parts.append("[OPTIONS]")
        
        # Add arguments
        for param in self.get_params(ctx):
            if isinstance(param, click.Argument):
                if param.required:
                    usage_parts.append(param.name.upper())
                else:
                    usage_parts.append(f"[{param.name.upper()}]")
        
        usage_str = " ".join(usage_parts)
        console.print(f"[text]Usage:[/text] [brand.bright]{usage_str}[/brand.bright]")
        console.print()


class RichGroup(click.Group):
    """Custom Click Group with Rich help formatting."""
    
    # Set command_class so all subcommands use Rich formatting
    command_class = RichCommand
    
    def format_help(self, ctx, formatter):
        """Format help output using Rich with styled organization."""
        # Write usage
        self.write_usage(ctx, formatter)
        
        # Write description
        if self.help:
            # Get first line of help as description
            desc_lines = self.help.split('\n')
            console.print(f"[text]{desc_lines[0].strip()}[/text]")
            console.print()
            console.print(f"[brand.bright]https://biolm.ai[/brand.bright]")
            console.print()
        
        # Organize commands into sections
        commands_by_section = {}
        for name, cmd in self.commands.items():
            # Determine section based on command name/type
            if name in ['login', 'logout', 'status', 'version']:
                section = 'Authentication'
            elif name == 'server':
                section = 'Server'
            elif name == 'workspace':
                section = 'Workspaces'
            elif name == 'model':
                section = 'Models'
            elif name == 'protocol':
                section = 'Protocols'
            elif name == 'dataset':
                section = 'Datasets'
            else:
                section = 'Commands'
            
            if section not in commands_by_section:
                commands_by_section[section] = []
            
            # If it's a group, expand to show subcommands
            if isinstance(cmd, click.Group) and cmd.commands:
                for sub_name, sub_cmd in sorted(cmd.commands.items()):
                    sub_help = sub_cmd.get_short_help_str() or sub_cmd.help or ''
                    # Format as "group subcommand"
                    full_name = f"{name} {sub_name}"
                    commands_by_section[section].append((full_name, sub_cmd))
            else:
                commands_by_section[section].append((name, cmd))
        
        # Write Options section with box
        opts = []
        for param in self.get_params(ctx):
            if isinstance(param, click.Option) and not param.hidden:
                rv = param.get_help_record(ctx)
                if rv:
                    opts.append(rv)
        if opts:
            # Create box content
            box_content = []
            for opt_name, opt_help in opts:
                # Format option name in brand color, description in text
                opt_padding = " " * max(0, 25 - len(opt_name))
                if opt_help:
                    line = f"[brand.bright]{opt_name}[/brand.bright]{opt_padding}  [text]{opt_help}[/text]"
                else:
                    line = f"[brand.bright]{opt_name}[/brand.bright]"
                box_content.append(line)
            
            # Create panel with box style
            panel = Panel(
                "\n".join(box_content),
                title="[bold]Options[/bold]",
                border_style="border",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            console.print(panel)
            console.print()
        
        # Write command sections in order with boxes
        section_order = ['Authentication', 'Server', 'Workspaces', 'Models', 'Protocols', 'Datasets', 'Commands']
        for section in section_order:
            if section in commands_by_section:
                # Create box content
                box_content = []
                for name, cmd in sorted(commands_by_section[section]):
                    help_text = cmd.get_short_help_str() or cmd.help or ''
                    # Format command name in brand color, description in text
                    cmd_padding = " " * max(0, 25 - len(name))
                    line = f"[brand.bright]{name}[/brand.bright]{cmd_padding}  [text]{help_text}[/text]"
                    box_content.append(line)
                
                # Create panel with box style
                panel = Panel(
                    "\n".join(box_content),
                    title=f"[bold]{section}[/bold]",
                    border_style="border",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
                console.print(panel)
                console.print()
    
    def write_usage(self, ctx, formatter):
        """Write usage line with Rich formatting."""
        console.print(f"[text]Usage:[/text] [brand.bright]{ctx.command_path}[/brand.bright] [OPTIONS] COMMAND [ARGS]...")
        console.print()


@click.command()
def main(args=None):
    """Console script for biolmai."""
    click.echo("Replace this message by putting your code into " "biolm.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


@click.group(cls=RichGroup, invoke_without_command=True)
@click.option("--debug/--no-debug", default=False)
@click.option(
    "--color/--no-color",
    default=None,
    help="Force color on or off (also respects NO_COLOR and BIOLM_CLI_THEME)",
)
@click.version_option(BIOLM_VERSION, prog_name="biolm", message="%(prog)s %(version)s")
def cli(debug, color):
    """BioLM CLI - Command-line interface for the BioLM platform.
    
    This CLI provides access to BioLM's biological language models and APIs.
    Use the commands below to authenticate, manage workspaces, run models,
    execute protocols, and work with datasets.
    """
    global console
    if color is False or (color is None and no_color_requested()):
        console = create_console(no_color=True)
    elif color is True:
        console = create_console(no_color=False)
    else:
        console = create_console()


@cli.command()
def version():
    """Show the installed biolm package version."""
    console.print(f"[brand.bright]biolm[/brand.bright] {BIOLM_VERSION}")


def _client_catalog_base() -> str:
    """Site root for model list/catalog (follows BIOLM_BASE_API_URL when set)."""
    return get_model_catalog_base()


def display_env_vars_table():
    """Display environment variables in a formatted Rich table."""
    table = Table(
        title="[brand]BioLM CLI Status[/brand]",
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bright",
    )
    table.add_column("Setting", style="brand", no_wrap=True)
    table.add_column("Value", style="text")

    for env_name in ("BIOLM_TOKEN", "BIOLMAI_TOKEN", "BIOLM_TOKEN"):
        env_var_tok = os.environ.get(env_name, "")
        if env_var_tok:
            masked = f"{env_var_tok[:6]}••••••••" if len(env_var_tok) >= 6 else "••••••••"
            table.add_row(env_name, masked)
        else:
            table.add_row(env_name, "[text.muted]Not set[/text.muted]")

    table.add_row("Credentials Path", str(ACCESS_TOK_PATH))
    table.add_row("Model API URL", BIOLM_BASE_API_URL)
    table.add_row("Platform Domain", BIOLM_BASE_DOMAIN)
    catalog_base = _client_catalog_base()
    if catalog_base.rstrip("/") != BIOLM_BASE_DOMAIN.rstrip("/"):
        table.add_row("Model Catalog Host", catalog_base)

    console.print(table)


@cli.command()
def status():
    """Show authentication status and configuration.
    
    Displays the current authentication status, including environment
    variables, credentials location, and API endpoint. Also validates
    existing credentials if present.
    """
    display_env_vars_table()
    console.print()  # Add spacing before auth status
    get_auth_status()


@cli.command()
@click.option(
    "--client-id",
    envvar="BIOLMAI_OAUTH_CLIENT_ID",
    default=None,
    help="OAuth client ID (defaults to ``BIOLMAI_PUBLIC_CLIENT_ID`` or ``BIOLMAI_OAUTH_CLIENT_ID`` env var)",
)
@click.option(
    "--scope",
    default="read write",
    show_default=True,
    help="OAuth scope string",
)
def login(client_id, scope):
    """Login to BioLM using OAuth 2.0 with PKCE.
    
    Checks for existing credentials and validates them. If credentials are missing
    or invalid, opens a browser for OAuth authorization. Credentials are saved to
    ``~/.biolmai/credentials``.
    
    Examples:

    .. code-block:: bash

        # Login with default client ID
        biolmai login

        # Login with custom client ID
        biolmai login --client-id your-client-id

        # Login with custom scope (supported: read, write, introspection)
        biolmai login --scope "read write"
    """
    # Check if credentials already exist and are valid
    if are_credentials_valid():
        console.print(Panel(
            "[success]✓ You are already logged in![/success]\n\n"
            f"Credentials: [brand]{ACCESS_TOK_PATH}[/brand]\n\n"
            "Run `biolm status` to view your authentication status.",
            title="[success]Authentication Status[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
        return
    
    # Use default client ID if not provided
    if not client_id:
        client_id = BIOLM_PUBLIC_CLIENT_ID
    
    if not client_id:
        console.print(Panel(
            "[error]✗ OAuth client ID required[/error]\n\n"
            "Set BIOLMAI_OAUTH_CLIENT_ID environment variable\n"
            "or pass --client-id",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        raise click.Abort()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Starting OAuth login...", total=None)
            console.print("A browser window will open for authorization.")
            progress.update(task, description="Waiting for browser authentication...")
            
            oauth_login(client_id=client_id, scope=scope)
            
            progress.update(task, description="[success]✓ Login successful![/success]")
        
        console.print()
        console.print(Panel(
            f"[success]✓ Login succeeded![/success]\n\n"
            f"Credentials saved to: [brand]{ACCESS_TOK_PATH}[/brand]",
            title="[success]Success[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    except Exception as e:
        console.print()
        console.print(Panel(
            f"[error]✗ Login failed[/error]\n\n[text]{str(e)}[/text]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        raise click.Abort()


@cli.command()
def logout():
    """Logout and remove saved credentials.
    
    Removes the saved authentication credentials from ``~/.biolmai/credentials``.
    After logout, you will need to run ``biolmai login`` again to authenticate.
    """
    try:
        os.remove(ACCESS_TOK_PATH)
        console.print("[success]✓ Successfully logged out[/success]")
    except FileNotFoundError:
        # File doesn't exist, user is already logged out - silently ignore
        console.print("[text.muted]Already logged out[/text.muted]")
    except Exception as e:
        console.print(f"[error]✗ Logout failed: {e}[/error]")
        raise click.Abort()


@cli.group(cls=RichGroup)
def server():
    """Run and manage the local biolm server proxy."""
    pass


@server.command("start")
@click.option("--host", default=None, help="Bind host (default: BIOLM_SERVER_HOST or 127.0.0.1)")
@click.option("--port", default=None, type=int, help="Bind port (default: BIOLM_SERVER_PORT or 8787)")
@click.option(
    "--auth",
    type=click.Choice(["none", "token"]),
    default=None,
    help="Server auth mode (default: BIOLM_SERVER_AUTH or none)",
)
@click.option(
    "--refresh-seconds",
    default=None,
    type=int,
    help="Registry refresh interval (default: BIOLM_SERVER_REFRESH_SECONDS or 60)",
)
@click.option(
    "--modal-env",
    default=None,
    help="Modal environment for this server (default: BIOLM_SERVER_MODAL_ENV or main)",
)
@click.option(
    "--detach",
    "-d",
    is_flag=True,
    help="Run the server in the background",
)
def server_start(host, port, auth, refresh_seconds, modal_env, detach):
    """Start the local biolm server proxy for Modal-deployed models."""
    try:
        from biolm.server.runner import SERVER_LOG_PATH, run_server_foreground, start_detached_server
        from biolm.server.settings import ServerSettings
    except ImportError:
        console.print(Panel(
            "[error]Server dependencies not installed.[/error]\n\n"
            "Install with: [brand]pip install biolm[server][/brand]",
            title="[error]Missing Dependencies[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        raise click.Abort()

    settings = ServerSettings.from_env(
        host=host,
        port=port,
        auth_mode=auth,
        refresh_seconds=refresh_seconds,
        modal_environment=modal_env,
    )
    try:
        settings.validate()
    except ValueError as exc:
        console.print(f"[error]{exc}[/error]")
        raise click.Abort()

    if detach:
        try:
            pid = start_detached_server(settings)
        except RuntimeError as exc:
            console.print(f"[error]{exc}[/error]")
            raise click.Abort()
        console.print(Panel(
            f"[success]biolm server started in background[/success]\n\n"
            f"URL: [brand]{settings.base_url}[/brand]\n"
            f"API: [brand]{settings.api_url}[/brand]\n"
            f"Auth: {settings.auth_mode}\n"
            f"Modal env: {settings.modal_environment}\n"
            f"PID: {pid}\n"
            f"Logs: [brand]{SERVER_LOG_PATH}[/brand]\n\n"
            f"Point model inference at this server:\n"
            f"  export BIOLM_BASE_API_URL={settings.api_url}\n\n"
            f"Stop with: [brand]biolm server stop[/brand]",
            title="[brand]biolm server[/brand]",
            border_style="brand",
            box=box.ROUNDED,
        ))
        return

    console.print(Panel(
        f"[success]Starting biolm server[/success]\n\n"
        f"URL: [brand]{settings.base_url}[/brand]\n"
        f"API: [brand]{settings.api_url}[/brand]\n"
        f"Auth: {settings.auth_mode}\n"
        f"Modal env: {settings.modal_environment}\n\n"
        f"Point model inference at this server:\n"
        f"  export BIOLM_BASE_API_URL={settings.api_url}\n\n"
        f"Press [brand]Ctrl+C[/brand] to stop, or use [brand]biolm server stop[/brand] from another terminal.",
        title="[brand]biolm server[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))
    run_server_foreground(settings)


@server.command("stop")
@click.option("--host", default=None, help="Server host (default: BIOLM_SERVER_HOST or 127.0.0.1)")
@click.option("--port", default=None, type=int, help="Server port (default: BIOLM_SERVER_PORT or 8787)")
@click.option("--force", is_flag=True, help="Send SIGKILL instead of SIGTERM")
def server_stop(host, port, force):
    """Stop a running biolm server."""
    from biolm.server.process import stop_server
    from biolm.server.settings import ServerSettings

    settings = ServerSettings.from_env(host=host, port=port)
    pid = stop_server(settings.host, settings.port, force=force)
    if not pid:
        console.print(
            f"[text.muted]No biolm server running at {settings.base_url}.[/text.muted]\n"
            f"[text.muted]Run [brand]biolm server start[/brand] to start the local proxy.[/text.muted]"
        )
        return

    verb = "Stopped" if not force else "Killed"
    console.print(f"[success]{verb} biolm server[/success] (pid {pid}, {settings.base_url})")


@server.command("status")
@click.option(
    "--modal-env",
    default=None,
    help="Modal environment to preview when the proxy is not running",
)
@click.option(
    "--limit",
    default=10,
    show_default=True,
    type=int,
    help="Max deployments to list (0 = show all)",
)
def server_status(modal_env, limit):
    """Show biolm server configuration, local proxy state, and deployments."""
    from biolm.server.process import fetch_server_health, get_server_runtime_status
    from biolm.server.registry import CompositeRegistry, ConfigRegistry, ModalRegistry
    from biolm.server.settings import ServerSettings

    settings = ServerSettings.from_env(modal_environment=modal_env)
    runtime = get_server_runtime_status(settings.host, settings.port)
    live_health = (
        fetch_server_health(settings.host, settings.port) if runtime.running else None
    )

    auth_mode = settings.auth_mode
    modal_environment = settings.modal_environment
    deployment_rows = []
    registry_source = "preview"

    if live_health:
        auth_mode = live_health.get("auth_mode", auth_mode)
        modal_environment = live_health.get("modal_environment", modal_environment)
        registry_source = "live"
        deployment_rows = sorted(
            live_health.get("deployments", []),
            key=lambda row: row.get("slug", ""),
        )
        if modal_env and modal_env != modal_environment:
            console.print(
                f"[text.muted]Proxy is running with Modal env [brand]{modal_environment}[/brand]; "
                f"ignoring --modal-env {modal_env}.[/text.muted]\n"
            )
    else:
        config_registry = ConfigRegistry(
            slugs=settings.configured_slugs(),
            config_path=settings.config_path,
        )
        modal_registry = ModalRegistry(environment_name=modal_environment)
        registry = CompositeRegistry(config_registry, modal_registry, health_check=False)

        import asyncio
        asyncio.run(registry.refresh())

        deployment_rows = [
            entry.to_dict() for entry in sorted(registry.list(), key=lambda e: e.slug)
        ]

    table = Table(title="[brand]biolm server status[/brand]", box=box.ROUNDED)
    table.add_column("Setting", style="brand")
    table.add_column("Value", style="text")
    if runtime.running:
        pid_text = str(runtime.pid) if runtime.pid else "[text.muted]unknown[/text.muted]"
        table.add_row("Local proxy", f"[success]running[/success] (pid {pid_text})")
    else:
        table.add_row("Local proxy", "[text.muted]not running[/text.muted]")
    table.add_row("URL", runtime.url)
    table.add_row("Host", settings.host)
    table.add_row("Port", str(settings.port))
    table.add_row("Auth mode", auth_mode)
    table.add_row(
        "Modal credentials",
        "yes" if ModalRegistry.modal_credentials_present() else "no",
    )
    table.add_row("Modal environment", modal_environment)
    table.add_row("Config path", settings.config_path)
    table.add_row("Models (env)", settings.models_env or "[text.muted]not set[/text.muted]")
    console.print(table)

    if registry_source == "live":
        console.print()
    else:
        console.print(
            "\n[text.muted]Deployments below are a Modal preview for this environment. "
            "Start the proxy with [brand]biolm server start --modal-env {env}[/brand].[/text.muted]\n".format(
                env=modal_environment
            )
        )

    total = len(deployment_rows)
    if not deployment_rows:
        console.print("[text.muted]No models in registry. Set BIOLM_SERVER_MODELS or ~/.biolm/server.yaml[/text.muted]")
        return

    shown = deployment_rows if limit == 0 else deployment_rows[: max(limit, 0)]
    if registry_source == "live":
        title = "Loaded deployments (live server)"
    else:
        title = "Discovered deployments (Modal preview)"
    if limit > 0 and total > limit:
        title = f"{title} — showing {len(shown)} of {total}"

    mtable = Table(title=title, box=box.ROUNDED)
    mtable.add_column("Slug", style="brand")
    mtable.add_column("Status")
    mtable.add_column("Source")
    mtable.add_column("URL", style="text.muted")
    for row in shown:
        mtable.add_row(
            row.get("slug", ""),
            row.get("status", "unknown"),
            row.get("source", ""),
            row.get("base_url", ""),
        )
    console.print(mtable)
    if limit > 0 and total > limit:
        console.print(
            f"\n[text.muted]Use [brand]--limit 0[/brand] to list all {total} deployments.[/text.muted]"
        )


@cli.group(cls=RichGroup)
def workspace():
    """Manage workspaces.
    
    Commands for creating, listing, and managing BioLM workspaces.
    """
    pass


@workspace.command()
def list():
    """List all workspaces.
    
    Display a list of all workspaces you have access to.
    """
    console.print(Panel(
        "[text.muted]Workspace commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to list and manage BioLM workspaces.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@workspace.command()
@click.argument('workspace_id', required=False)
def show(workspace_id):
    """Show workspace details.
    
    Display information about a specific workspace. If no workspace ID is provided,
    shows information about the current workspace.
    """
    console.print(Panel(
        "[text.muted]Workspace commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to manage BioLM workspaces.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@workspace.command()
@click.argument('name')
def create(name):
    """Create a new workspace.
    
    Create a new workspace with the specified name.
    """
    console.print(Panel(
        "[text.muted]Workspace commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to create BioLM workspaces.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@workspace.command()
@click.argument('workspace_id')
def delete(workspace_id):
    """Delete a workspace.
    
    Delete the specified workspace.
    """
    console.print(Panel(
        "[text.muted]Workspace commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to delete BioLM workspaces.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


# Helper functions for model commands
def _format_tags(tags: List[str]) -> str:
    """Format tags with emoji and color."""
    if not tags:
        return "[text.muted]—[/text.muted]"
    # Show first 5 tags, color each tag
    tag_colors = ["accent", "brand", "success", "warning", "brand.bright"]
    formatted_tags = []
    for i, tag in enumerate(tags[:5]):
        color = tag_colors[i % len(tag_colors)]
        formatted_tags.append(f"[{color}]🏷️ {tag}[/{color}]")
    result = " ".join(formatted_tags)
    if len(tags) > 5:
        result += f" [text.muted](+{len(tags) - 5} more)[/text.muted]"
    return result


def _format_actions(actions: List[str]) -> str:
    """Format actions with emojis and colors."""
    if not actions:
        return "[text.muted]—[/text.muted]"
    
    action_emojis = {
        'encode': '🔢',
        'predict': '🔮',
        'generate': '✨',
        'classify': '🏷️',
        'similarity': '🔍',
        'lookup': '🔎',
    }
    
    action_colors = {
        'encode': 'brand',
        'predict': 'success',
        'generate': 'accent',
        'classify': 'warning',
        'similarity': 'brand.bright',
        'lookup': 'text',
    }
    
    formatted = []
    for action in actions:
        emoji = action_emojis.get(action, '⚡')
        color = action_colors.get(action, 'text')
        formatted.append(f"[{color}]{emoji} {action}[/{color}]")
    
    return " ".join(formatted)


def _format_description(description: str, max_length: int = 100) -> str:
    """Format description with emoji and color."""
    if not description:
        return "[text.muted]—[/text.muted]"
    
    truncated = description
    if len(description) > max_length:
        truncated = description[:max_length - 3] + "..."
    
    return f"[text]📝 {truncated}[/text]"


def _format_date(date_str: str, include_emoji: bool = True) -> str:
    """Format date with emoji and color."""
    if not date_str:
        return "[text.muted]—[/text.muted]"
    
    # Try to parse and format date
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        formatted_date = dt.strftime('%Y-%m-%d')
        emoji = "📅 " if include_emoji else ""
        return f"[text.muted]{emoji}{formatted_date}[/text.muted]"
    except:
        emoji = "📅 " if include_emoji else ""
        return f"[text.muted]{emoji}{date_str}[/text.muted]"


def _format_capability(value: bool, label: str) -> str:
    """Format capability (encoder, predictor, etc.) with emoji."""
    if value:
        return f"[success]✅ {label}[/success]"
    else:
        return f"[text.muted]❌ {label}[/text.muted]"


def _parse_filter_expression(filter_str: str) -> tuple[str, Any]:
    """Parse filter expression like 'encoder=true' or 'model_name=esm2'.
    
    Args:
        filter_str: Filter expression string (e.g., 'encoder=true', 'model_name=esm2')
        
    Returns:
        Tuple of (field_name, expected_value)
        
    Raises:
        ValueError: If filter expression is invalid
    """
    if '=' not in filter_str:
        raise ValueError(f"Invalid filter expression: {filter_str}. Expected format: field=value")
    
    field, value_str = filter_str.split('=', 1)
    field = field.strip()
    value_str = value_str.strip()
    
    # Try to convert value to appropriate type
    if value_str.lower() == 'true':
        value = True
    elif value_str.lower() == 'false':
        value = False
    elif value_str.lower() == 'null' or value_str.lower() == 'none':
        value = None
    elif value_str.isdigit():
        value = int(value_str)
    else:
        # Try float
        try:
            value = float(value_str)
        except ValueError:
            # Keep as string
            value = value_str
    
    return field, value


def _filter_models(models: List[Dict], filter_expr: str) -> List[Dict]:
    """Filter model list based on filter expression.
    
    Args:
        models: List of model dictionaries
        filter_expr: Filter expression (e.g., 'encoder=true')
        
    Returns:
        Filtered list of models
    """
    if not filter_expr:
        return models
    
    field, expected_value = _parse_filter_expression(filter_expr)
    
    filtered = []
    for model in models:
        # Handle both old and new API response formats
        model_value = model.get(field)
        if model_value is None:
            # Try alternative field names
            if field == 'model_name':
                model_value = model.get('name')
            elif field == 'model_slug':
                model_value = model.get('slug')
        
        if model_value == expected_value:
            filtered.append(model)
    
    return filtered


def _sort_models(models: List[Dict], sort_field: str) -> List[Dict]:
    """Sort model list by field.
    
    Args:
        models: List of model dictionaries
        sort_field: Field name to sort by (optionally prefixed with '-' for descending)
        
    Returns:
        Sorted list of models
    """
    if not sort_field:
        return models
    
    # Check for descending sort
    descending = False
    if sort_field.startswith('-'):
        descending = True
        sort_field = sort_field[1:]
    
    def get_sort_value(model: dict) -> Any:
        """Get sort value from model, handling various field names."""
        value = model.get(sort_field)
        if value is None:
            # Try alternative field names
            if sort_field == 'model_name':
                value = model.get('name')
            elif sort_field == 'model_slug':
                value = model.get('slug')
        
        # Handle None values (put at end)
        if value is None:
            return '' if not descending else 'zzz'
        
        # Convert to comparable type
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        return str(value).lower()
    
    return sorted(models, key=get_sort_value, reverse=descending)


def _detect_file_format(file_path: Union[str, Path]) -> str:
    """Detect file format from extension.
    
    Args:
        file_path: Path to file or file-like object
        
    Returns:
        Format string: 'fasta', 'csv', 'pdb', 'json', or 'unknown'
    """
    # Handle file-like objects (StringIO, etc.) - can't detect from extension
    if hasattr(file_path, 'read') and not hasattr(file_path, 'suffix'):
        return 'unknown'
    
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Check if it's a Path object with suffix
    if not hasattr(file_path, 'suffix'):
        return 'unknown'
    
    ext = file_path.suffix.lower()
    
    if ext in ['.fasta', '.fa', '.fas']:
        return 'fasta'
    elif ext == '.csv':
        return 'csv'
    elif ext == '.pdb':
        return 'pdb'
    elif ext in ['.json', '.jsonl']:
        return 'json'
    else:
        return 'unknown'


def _load_input_data(file_path: Union[str, Path], format: Optional[str] = None, type: Optional[str] = None) -> List[Dict]:
    """Load input data from file using appropriate IO module.
    
    Args:
        file_path: Path to input file or '-' for stdin
        format: Format override ('fasta', 'csv', 'pdb', 'json')
        type: Input type override (for API requests)
        
    Returns:
        List of dictionaries ready for API requests
        
    Raises:
        ValueError: If format cannot be determined or file cannot be loaded
        FileNotFoundError: If file doesn't exist
    """
    # Handle stdin
    if file_path == '-' or (isinstance(file_path, str) and file_path == '-'):
        if not format:
            raise ValueError("Format must be specified when reading from stdin (use --format)")
        file_path = sys.stdin
    else:
        # Auto-detect format if not provided
        if not format:
            format = _detect_file_format(file_path)
            if format == 'unknown':
                raise ValueError(
                    f"Cannot detect file format from extension. "
                    f"Please specify --format option. "
                    f"Supported formats: fasta, csv, pdb, json"
                )
    
    # Load data based on format
    # CRITICAL: Validate format before loading to prevent JSON files being read as FASTA
    if format == 'json':
        data = load_json(file_path)
        # Validate that we got proper dicts, not strings
        if data and isinstance(data[0], str):
            raise ValueError(
                f"JSON file appears to be parsed incorrectly. "
                f"This usually means format detection failed. "
                f"File: {file_path}, Detected format: {format}. "
                f"Try specifying --format json explicitly."
            )
    elif format == 'fasta':
        data = load_fasta(file_path)
        # Convert to API format if type is specified
        if type:
            data = [{type: item.get('sequence', '')} for item in data]
    elif format == 'csv':
        data = load_csv(file_path)
        # Convert to API format if type is specified
        if type:
            # Assume first column or 'sequence' column contains the data
            for item in data:
                if type not in item:
                    # Try to find sequence-like data
                    if 'sequence' in item:
                        item[type] = item.pop('sequence')
                    else:
                        # Use first value
                        first_key = next(iter(item.keys()), None)
                        if first_key:
                            item[type] = item.pop(first_key)
    elif format == 'pdb':
        data = load_pdb(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported: json, fasta, csv, pdb")
    
    return data


def _save_output_data(data: List[Dict], file_path: Optional[Union[str, Path]], format: Optional[str] = None) -> None:
    """Save output data to file using appropriate IO module.
    
    Args:
        data: List of dictionaries from API response
        file_path: Path to output file, None for stdout, or '-' for stdout
        format: Format override ('json', 'fasta', 'csv', 'pdb')
        
    Raises:
        ValueError: If format cannot be determined
    """
    # Determine format
    if file_path and file_path != '-':
        if not format:
            format = _detect_file_format(file_path)
            if format == 'unknown':
                # Default to JSON
                format = 'json'
    else:
        # stdout - default to JSON
        if not format:
            format = 'json'
        file_path = '-'
    
    # Save data based on format
    if format == 'json':
        # Check if JSONL based on extension
        jsonl = False
        if file_path != '-' and isinstance(file_path, (str, Path)):
            if Path(file_path).suffix == '.jsonl':
                jsonl = True
        to_json(data, file_path, jsonl=jsonl)
    elif format == 'fasta':
        to_fasta(data, file_path)
    elif format == 'csv':
        to_csv(data, file_path)
    elif format == 'pdb':
        to_pdb(data, file_path)
    else:
        raise ValueError(f"Unsupported output format: {format}")


@cli.group(cls=RichGroup)
def model():
    """Work with BioLM models.
    
    Commands for listing available models, viewing model details, and running models.
    """
    pass


@model.command()
@click.option('--filter', help='Filter models (e.g., encoder=true, model_name=esm2)')
@click.option('--sort', help='Sort by field (e.g., model_name, -model_name for descending)')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml', 'csv']), default='table', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Save output to file')
@click.option('--fields', help='Comma-separated list of fields to display')
@click.option('--view', type=click.Choice(['compact', 'detailed', 'full', 'enriched']), help='Predefined field views (enriched includes description, tags, etc.)')
def list(filter, sort, format, output, fields, view):
    """List available models.
    
    Display a list of all available BioLM models with filtering, sorting, and
    various output format options.
    
    Examples:

    .. code-block:: bash

        # List all models
        biolmai model list

        # Filter for encoder models
        biolmai model list --filter encoder=true

        # Sort by model name
        biolmai model list --sort model_name

        # Output as JSON
        biolmai model list --format json --output models.json

        # Compact view
        biolmai model list --view compact
    """
    try:
        with console.status("[brand]Fetching models...[/brand]"):
            models = list_models(base_url=_client_catalog_base())
        
        if not models:
            console.print(Panel(
                "[error]No models found.[/error]\n\n"
                "Please check your connection and authentication.",
                title="[error]Error[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Apply filtering
        if filter:
            try:
                models = _filter_models(models, filter)
            except ValueError as e:
                console.print(Panel(
                    f"[error]{str(e)}[/error]",
                    title="[error]Invalid Filter[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
        
        # Apply sorting
        if sort:
            models = _sort_models(models, sort)
        
        if not models:
            console.print("[text.muted]No models match the specified filter.[/text.muted]")
            sys.exit(0)
        
        # Determine fields to display
        if view:
            if view == 'compact':
                default_fields = ['model_name', 'model_slug', 'actions']
            elif view == 'detailed':
                default_fields = ['model_name', 'model_slug', 'actions', 'encoder', 'predictor', 'generator']
            elif view == 'enriched':
                default_fields = ['model_name', 'model_slug', 'actions', 'description', 'tags', 'created_at']
            else:  # full
                default_fields = None  # Show all fields
        else:
            default_fields = ['model_name', 'model_slug', 'actions']
        
        if fields:
            display_fields = [f.strip() for f in fields.split(',')]
        else:
            display_fields = default_fields
        
        # Output based on format
        if format == 'json':
            output_data = json.dumps(models, indent=2, default=str)
            if output:
                with open(output, 'w') as f:
                    f.write(output_data)
                console.print(f"[success]✓ Models saved to {output}[/success]")
            else:
                console.print(output_data, markup=False, highlight=False)
        elif format == 'yaml':
            try:
                import yaml
                output_data = yaml.dump(models, default_flow_style=False, allow_unicode=True)
                if output:
                    with open(output, 'w') as f:
                        f.write(output_data)
                    console.print(f"[success]✓ Models saved to {output}[/success]")
                else:
                    console.print(output_data)
            except ImportError:
                console.print(Panel(
                    "[error]PyYAML is required for YAML output.[/error]\n\n"
                    "Install with: pip install pyyaml",
                    title="[error]Missing Dependency[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
        elif format == 'csv':
            if not models:
                console.print("[text.muted]No models to display.[/text.muted]")
                sys.exit(0)
            
            # Flatten models for CSV
            csv_data = []
            for model in models:
                row = {}
                # Handle both old and new API formats
                for field in display_fields if display_fields else model.keys():
                    value = model.get(field)
                    if value is None:
                        # Try alternative field names
                        if field == 'model_name':
                            value = model.get('name')
                        elif field == 'model_slug':
                            value = model.get('slug')
                        elif field == 'actions':
                            # Build actions list
                            actions = []
                            if 'actions' in model and isinstance(model['actions'], builtins.list):
                                actions = model['actions']
                            else:
                                if model.get('encoder'):
                                    actions.append('encode')
                                if model.get('predictor'):
                                    actions.append('predict')
                                if model.get('generator'):
                                    actions.append('generate')
                                if model.get('classifier'):
                                    actions.append('classify')
                                if model.get('similarity'):
                                    actions.append('similarity')
                            value = ', '.join(actions) if actions else ''
                    
                    # Convert value to string
                    if isinstance(value, (builtins.list, builtins.dict)):
                        value = json.dumps(value)
                    elif value is None:
                        value = ''
                    else:
                        value = str(value)
                    
                    row[field] = value
                csv_data.append(row)
            
            to_csv(csv_data, output if output else '-')
            if output:
                console.print(f"[success]✓ Models saved to {output}[/success]")
        else:  # table format
            table = Table(
                title="[brand]🤖 Available BioLM Models[/brand]",
                show_header=True,
                header_style="brand.bold",
                box=box.ROUNDED,
                title_style="brand.bright",
            )
            
            # Add columns based on display_fields with emojis
            emoji_map = {
                'model_name': '🤖',
                'model_slug': '🔗',
                'actions': '⚡',
                'description': '📝',
                'tags': '🏷️',
                'created_at': '📅',
                'encoder': '🔢',
                'predictor': '🔮',
                'generator': '✨',
            }
            if display_fields:
                for field in display_fields:
                    # Use friendly column names with emojis
                    emoji = emoji_map.get(field, '')
                    col_name = f"{emoji} {field.replace('_', ' ').title()}" if emoji else field.replace('_', ' ').title()
                    table.add_column(col_name, style="text")
            else:
                # Show all available fields (limit to common ones)
                common_fields = ['model_name', 'model_slug', 'actions', 'encoder', 'predictor', 'generator']
                for field in common_fields:
                    col_name = field.replace('_', ' ').title()
                    table.add_column(col_name, style="text")
            
            # Add rows
            for model in models[:100]:  # Limit to first 100 for display
                row_data = []
                
                fields_to_use = display_fields if display_fields else ['model_name', 'model_slug', 'actions']
                for field in fields_to_use:
                    value = model.get(field)
                    if value is None:
                        # Try alternative field names
                        if field == 'model_name':
                            value = model.get('name') or 'Unknown'
                        elif field == 'model_slug':
                            value = model.get('slug') or 'N/A'
                        elif field == 'actions':
                            # Build actions list
                            actions = []
                            if 'actions' in model and isinstance(model['actions'], builtins.list):
                                actions = model['actions']
                            else:
                                if model.get('encoder'):
                                    actions.append('encode')
                                if model.get('predictor'):
                                    actions.append('predict')
                                if model.get('generator'):
                                    actions.append('generate')
                                if model.get('classifier'):
                                    actions.append('classify')
                                if model.get('similarity'):
                                    actions.append('similarity')
                            value = ', '.join(actions) if actions else 'N/A'
                        else:
                            value = 'N/A'
                    
                    # Format value for display with colors and emojis
                    if field == 'tags' and isinstance(value, builtins.list):
                        value = _format_tags(value)
                    elif field == 'actions':
                        # Actions might be a list or we need to build it
                        if isinstance(value, builtins.list):
                            value = _format_actions(value)
                        else:
                            # Build actions list from boolean flags if needed
                            actions_list = []
                            if 'actions' in model and isinstance(model['actions'], builtins.list):
                                actions_list = model['actions']
                            else:
                                if model.get('encoder'):
                                    actions_list.append('encode')
                                if model.get('predictor'):
                                    actions_list.append('predict')
                                if model.get('generator'):
                                    actions_list.append('generate')
                                if model.get('classifier'):
                                    actions_list.append('classify')
                                if model.get('similarity'):
                                    actions_list.append('similarity')
                            value = _format_actions(actions_list)
                    elif field == 'description' and isinstance(value, str):
                        value = _format_description(value)
                    elif field == 'created_at' and isinstance(value, str):
                        value = _format_date(value)
                    elif field in ['encoder', 'predictor', 'generator', 'classifier', 'similarity'] and isinstance(value, bool):
                        value = _format_capability(value, field.replace('_', ' ').title())
                    elif isinstance(value, bool):
                        value = '[success]✓[/success]' if value else '[text.muted]✗[/text.muted]'
                    elif isinstance(value, (builtins.list, builtins.dict)):
                        value = json.dumps(value)
                    elif value is None:
                        value = '[text.muted]—[/text.muted]'
                    else:
                        # Default: just convert to string
                        value = str(value)
                    
                    row_data.append(value)
                
                table.add_row(*row_data)
            
            if len(models) > 100:
                table.add_row(*(['...'] * len(fields_to_use)))
                console.print(f"\n[text.muted]Showing first 100 of {len(models)} models. Use --filter to narrow results.[/text.muted]")
            
            console.print(table)
            if output:
                # Also save table data to file as JSON
                with open(output, 'w') as f:
                    json.dump(models, f, indent=2, default=str)
                console.print(f"\n[success]✓ Model data saved to {output}[/success]")
    
    except Exception as e:
        console.print(Panel(
            f"[error]Error listing models: {e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        if hasattr(e, '__cause__') and e.__cause__:
            console.print(f"[text.muted]{e.__cause__}[/text.muted]")
        sys.exit(1)


@model.command("catalog")
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Save output to file')
def catalog(format, output):
    """List the full OSS model catalog (all deployable models).

    Unlike ``model list``, this shows every model in the bundled OSS catalog,
    not only models deployed on the current server.
    """
    try:
        from biolm.server.catalog import list_catalog_models
        models = list_catalog_models()
    except ImportError:
        console.print(Panel(
            "[error]Catalog unavailable. Install biolm[server] or use hosted platform.[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)

    if format == 'json':
        payload = json.dumps(models, indent=2, default=str)
        if output:
            with open(output, 'w') as f:
                f.write(payload)
            console.print(f"[success]✓ Catalog saved to {output}[/success]")
        else:
            console.print(payload, markup=False, highlight=False)
    elif format == 'yaml':
        import yaml
        payload = yaml.dump(models, default_flow_style=False)
        if output:
            with open(output, 'w') as f:
                f.write(payload)
            console.print(f"[success]✓ Catalog saved to {output}[/success]")
        else:
            console.print(payload, markup=False, highlight=False)
    else:
        table = Table(title="[brand]OSS Model Catalog[/brand]", box=box.ROUNDED)
        table.add_column("Slug", style="brand")
        table.add_column("Name")
        table.add_column("Actions")
        for m in models:
            slug = m.get('model_slug') or m.get('slug', '')
            name = m.get('model_name') or m.get('name', '')
            actions = m.get('actions') or []
            table.add_row(slug, name, ', '.join(actions) if actions else '-')
        console.print(table)


@model.command()
@click.argument('model_name')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Save output to file')
@click.option('--include-schemas', is_flag=True, help='Include JSON schemas for each action')
@click.option('--include-code-examples', is_flag=True, help='Include code examples from API (fetches detailed model info)')
def show(model_name, format, output, include_schemas, include_code_examples):
    """Show model details.
    
    Display detailed information about a specific model, including metadata,
    available actions, and optionally JSON schemas for each action.
    
    Examples:

    .. code-block:: bash

        # Show model details
        biolmai model show esm2-8m

        # Include schemas
        biolmai model show esmfold --include-schemas

        # Output as JSON
        biolmai model show esm2-8m --format json --output model.json
    """
    try:
        with console.status("[brand]Fetching model information...[/brand]"):
            models = list_models(base_url=_client_catalog_base())
        
        if not models:
            console.print(Panel(
                "[error]Could not fetch models.[/error]\n\n"
                "Please check your connection and authentication.",
                title="[error]Error[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Find model by name or slug
        model_info = None
        for model in models:
            model_slug = model.get('model_slug') or model.get('slug')
            model_name_field = model.get('model_name') or model.get('name')
            if (model_slug and model_slug == model_name) or (model_name_field and model_name_field == model_name):
                model_info = model
                break
        
        if not model_info:
            console.print(Panel(
                f"[error]Model '{model_name}' not found.[/error]\n\n"
                f"Use 'biolm model list' to see available models.",
                title="[error]Model Not Found[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Extract available actions
        actions = []
        if 'actions' in model_info and isinstance(model_info['actions'], builtins.list):
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
        
        # Fetch schemas if requested
        schemas = {}
        if include_schemas and actions:
            from biolm.core.http import BioLMApiClient
            import asyncio
            
            async def fetch_schemas():
                client = BioLMApiClient(
                    model_info.get('model_slug') or model_info.get('slug') or model_name,
                    raise_httpx=False
                )
                try:
                    for action in actions:
                        schema = await client.schema(
                            model_info.get('model_slug') or model_info.get('slug') or model_name,
                            action
                        )
                        if schema:
                            schemas[action] = schema
                finally:
                    await client.shutdown()
            
            with console.status("[brand]Fetching schemas...[/brand]"):
                asyncio.run(fetch_schemas())
        
        # Fetch detailed model information if requested
        detailed_info = None
        if include_code_examples:
            model_slug = model_info.get('model_slug') or model_info.get('slug') or model_name
            with console.status("[brand]Fetching detailed model information...[/brand]"):
                detailed_info = get_model_details(
                    model_slug,
                    code_examples=True,
                    exclude_docs_html=True,
                    base_url=_client_catalog_base(),
                )
        
        # Merge detailed info with basic info if available
        if detailed_info:
            # Merge detailed info, giving priority to detailed_info
            model_info = {**model_info, **detailed_info}
        
        # Prepare output data
        output_data = {
            'model_name': model_info.get('model_name') or model_info.get('name'),
            'model_slug': model_info.get('model_slug') or model_info.get('slug'),
            'actions': actions,
            'metadata': {k: v for k, v in model_info.items() 
                        if k not in ['model_name', 'name', 'model_slug', 'slug', 'actions']}
        }
        
        if include_schemas and schemas:
            output_data['schemas'] = schemas
        
        # Add code examples if available
        if include_code_examples and detailed_info:
            if 'code_examples' in detailed_info:
                output_data['code_examples'] = detailed_info['code_examples']
        
        # Output based on format
        if format == 'json':
            output_str = json.dumps(output_data, indent=2, default=str)
            if output:
                with open(output, 'w') as f:
                    f.write(output_str)
                console.print(f"[success]✓ Model information saved to {output}[/success]")
            else:
                console.print(output_str, markup=False, highlight=False)
        elif format == 'yaml':
            try:
                import yaml
                output_str = yaml.dump(output_data, default_flow_style=False, allow_unicode=True)
                if output:
                    with open(output, 'w') as f:
                        f.write(output_str)
                    console.print(f"[success]✓ Model information saved to {output}[/success]")
                else:
                    console.print(output_str)
            except ImportError:
                console.print(Panel(
                    "[error]PyYAML is required for YAML output.[/error]\n\n"
                    "Install with: pip install pyyaml",
                    title="[error]Missing Dependency[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
        else:  # table format
            # Display model information in panels
            model_name_display = output_data['model_name'] or model_name
            model_slug_display = output_data['model_slug'] or 'N/A'
            
            # Main model info panel with emojis
            info_lines = [
                f"[bold]🤖 Name:[/bold] [brand]{model_name_display}[/brand]",
                f"[bold]🔗 Slug:[/bold] [text]{model_slug_display}[/text]",
            ]
            
            if actions:
                formatted_actions = _format_actions(actions)
                info_lines.append(f"[bold]Actions:[/bold] {formatted_actions}")
            
            # Add other metadata with enhanced formatting
            metadata = output_data.get('metadata', {})
            if metadata:
                for key, value in sorted(metadata.items()):
                    if value is not None and key not in ['encoder', 'predictor', 'generator', 'classifier', 'similarity']:
                        if key == 'tags' and isinstance(value, builtins.list):
                            value_str = _format_tags(value)
                        elif key == 'description' and isinstance(value, str):
                            value_str = _format_description(value, max_length=200)  # Longer for show command
                        elif key == 'created_at' and isinstance(value, str):
                            value_str = _format_date(value, include_emoji=False)  # Emoji already in label
                        elif isinstance(value, bool):
                            value_str = '[success]✓[/success]' if value else '[text.muted]✗[/text.muted]'
                        elif isinstance(value, (builtins.list, builtins.dict)):
                            value_str = json.dumps(value)
                        else:
                            value_str = str(value)
                        
                        # Add emoji prefix for certain fields
                        emoji_map = {
                            'description': '📝',
                            'tags': '🏷️',
                            'created_at': '📅',
                            'api_docs_link': '🔗',
                            'docs_link': '📚',
                        }
                        emoji = emoji_map.get(key, '')
                        field_label = f"{emoji} {key.replace('_', ' ').title()}" if emoji else key.replace('_', ' ').title()
                        info_lines.append(f"[bold]{field_label}:[/bold] {value_str}")
            
            console.print(Panel(
                "\n".join(info_lines),
                title=f"[brand]🤖 {model_name_display}[/brand]",
                border_style="brand",
                box=box.ROUNDED,
            ))
            
            # Display schemas if included
            if include_schemas and schemas:
                console.print()
                for action, schema in schemas.items():
                    schema_str = json.dumps(schema, indent=2)
                    console.print(Panel(
                        schema_str,
                        title=f"[brand]Schema: {action}[/brand]",
                        border_style="border",
                        box=box.ROUNDED,
                    ))
            elif include_schemas and not schemas:
                console.print()
                console.print("[text.muted]No schemas available for this model.[/text.muted]")
            
            # Display code examples if included
            if include_code_examples and 'code_examples' in output_data:
                console.print()
                code_examples = output_data['code_examples']
                if isinstance(code_examples, dict):
                    for action, example_code in code_examples.items():
                        if example_code:
                            console.print(Panel(
                                example_code,
                                title=f"[brand]Code Example: {action}[/brand]",
                                border_style="border",
                                box=box.ROUNDED,
                            ))
                elif isinstance(code_examples, str):
                    console.print(Panel(
                        code_examples,
                        title="[brand]Code Examples[/brand]",
                        border_style="border",
                        box=box.ROUNDED,
                    ))
            elif include_code_examples and 'code_examples' not in output_data:
                console.print()
                console.print("[text.muted]No code examples available for this model.[/text.muted]")
            
            if output:
                # Also save to file as JSON
                with open(output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                console.print(f"\n[success]✓ Model information saved to {output}[/success]")
    
    except Exception as e:
        console.print(Panel(
            f"[error]Error showing model details: {e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        if hasattr(e, '__cause__') and e.__cause__:
            console.print(f"[text.muted]{e.__cause__}[/text.muted]")
        sys.exit(1)


@model.command()
@click.argument('model_name')
@click.argument('action', type=click.Choice(['encode', 'predict', 'generate', 'lookup']))
@click.option('--input', '-i', type=click.Path(exists=False), help='Input file path or "-" for stdin')
@click.option('--output', '-o', type=click.Path(), help='Output file path (default: stdout)')
@click.option('--format', type=click.Choice(['json', 'fasta', 'csv', 'pdb']), help='Output format (auto-detected from output file extension)')
@click.option('--input-format', type=click.Choice(['json', 'fasta', 'csv', 'pdb']), help='Input format (auto-detected from input file extension)')
@click.option('--type', help='Input type override (sequence, pdb, context, etc.)')
@click.option('--params', help='Parameters as JSON string or file path')
@click.option('--batch-size', type=int, help='Batch size for processing (default: auto-detect from schema)')
@click.option('--progress', is_flag=True, help='Show progress bar for batch processing')
def run(model_name, action, input, output, format, input_format, type, params, batch_size, progress):
    """Run a model.
    
    Execute a BioLM model with the specified action. Supports reading from files
    (FASTA, CSV, PDB, JSON) or stdin, and writing results to files or stdout.
    
    Examples:

    .. code-block:: bash

        # Run model with inline input
        echo '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}' | biolmai model run esm2-8m encode -i - --format json

        # Run model with FASTA file
        biolmai model run esmfold predict -i sequences.fasta -o results.json

        # Run with parameters
        biolmai model run esm2-8m encode -i seq.fasta --params '{"normalize": true}'

        # Run with progress bar
        biolmai model run esmfold predict -i large.fasta --progress
    """
    # Initialize items variable for error reporting
    items = None
    
    try:
        # Load parameters if provided
        params_dict = None
        if params:
            if Path(params).exists():
                # Load from file
                with open(params, 'r') as f:
                    params_dict = json.load(f)
            else:
                # Parse as JSON string
                try:
                    params_dict = json.loads(params)
                except json.JSONDecodeError as e:
                    console.print(Panel(
                        f"[error]Invalid JSON in --params: {e}[/error]",
                        title="[error]Invalid Parameters[/error]",
                        border_style="error",
                        box=box.ROUNDED,
                    ))
                    sys.exit(1)
        
        # Show initial feedback
        console.print(f"[brand]🤖 Running {model_name} {action}...[/brand]")
        
        # Load input data
        if input:
            if input == '-':
                # Read from stdin
                # For stdin, need input format (not output format)
                stdin_input_format = input_format if input_format else format  # Fallback to format if input_format not specified
                if not stdin_input_format:
                    console.print(Panel(
                        "[error]Format must be specified when reading from stdin.[/error]\n\n"
                        "Use --input-format json, fasta, csv, or pdb (or --format as fallback)",
                        title="[error]Format Required[/error]",
                        border_style="error",
                        box=box.ROUNDED,
                    ))
                    sys.exit(1)
                
                with console.status("[brand]Reading from stdin...[/brand]"):
                    # Read stdin content
                    stdin_content = sys.stdin.read()
                    if not stdin_content.strip():
                        console.print(Panel(
                            "[error]No input data provided from stdin.[/error]",
                            title="[error]Empty Input[/error]",
                            border_style="error",
                            box=box.ROUNDED,
                        ))
                        sys.exit(1)
                
                # Create temporary file-like object
                import io
                file_obj = io.StringIO(stdin_content)
                items = _load_input_data(file_obj, format=stdin_input_format, type=type)
            else:
                # Load from file
                # Use input_format if specified, otherwise auto-detect
                input_format_to_use = input_format
                # CRITICAL: Force JSON format for .json/.jsonl files FIRST - before ANY detection
                if isinstance(input, str) and (input.endswith('.json') or input.endswith('.jsonl')):
                    input_format_to_use = 'json'
                    console.print(f"[text.muted]Using JSON format for .json file[/text.muted]")
                elif not input_format_to_use:
                    detected_format = _detect_file_format(input)
                    if detected_format == 'unknown':
                        # Try to detect from file content as fallback
                        try:
                            with open(input, 'r') as f:
                                first_chars = f.read(100)
                                if first_chars.strip().startswith('[') or first_chars.strip().startswith('{'):
                                    detected_format = 'json'
                                    console.print(f"[text.muted]Auto-detected JSON format from content[/text.muted]")
                                else:
                                    console.print(Panel(
                                        f"[error]Cannot detect file format from extension or content.[/error]\n\n"
                                        f"File: {input}\n"
                                        f"Please specify --input-format option.\n"
                                        f"Supported formats: json, fasta, csv, pdb",
                                        title="[error]Format Detection Failed[/error]",
                                        border_style="error",
                                        box=box.ROUNDED,
                                    ))
                                    sys.exit(1)
                        except Exception:
                            console.print(Panel(
                                f"[error]Cannot detect file format from extension.[/error]\n\n"
                                f"File: {input}\n"
                                f"Please specify --input-format option.\n"
                                f"Supported formats: json, fasta, csv, pdb",
                                title="[error]Format Detection Failed[/error]",
                                border_style="error",
                                box=box.ROUNDED,
                            ))
                            sys.exit(1)
                    input_format_to_use = detected_format
                    console.print(f"[text.muted]Detected input format: {input_format_to_use}[/text.muted]")
                
                # CRITICAL: Force JSON format for .json files regardless of detection
                if isinstance(input, str) and (input.endswith('.json') or input.endswith('.jsonl')):
                    if input_format_to_use != 'json':
                        console.print(f"[warning]Forcing JSON format for .json file (was: {input_format_to_use})[/warning]")
                    input_format_to_use = 'json'
                
                with console.status(f"[brand]Loading input from {input}...[/brand]"):
                    items = _load_input_data(input, format=input_format_to_use, type=type)
        else:
            # No input specified - this is an error for run command
            console.print(Panel(
                "[error]Input is required.[/error]\n\n"
                "Specify input with --input/-i option or use '-' for stdin.\n\n"
                "Examples:\n"
                "  biolm model run esm2-8m encode -i sequences.fasta\n"
                "  echo '{\"sequence\": \"ACDEF\"}' | biolm model run esm2-8m encode -i - --format json",
                title="[error]Missing Input[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        if not items:
            console.print(Panel(
                "[error]No items loaded from input.[/error]",
                title="[error]Empty Input[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Validate items structure
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                # Check if it's a string that looks like JSON (common error - file read as FASTA)
                if isinstance(item, str) and (item.strip().startswith('[') or item.strip().startswith('{')):
                    console.print(Panel(
                        f"[error]Item {i+1} appears to be unparsed JSON string.[/error]\n\n"
                        f"This usually means the file format was detected incorrectly.\n"
                        f"File: {input if 'input' in locals() else 'unknown'}\n"
                        f"Detected format: {format if 'format' in locals() else 'unknown'}\n\n"
                        f"Try specifying format explicitly: --format json\n\n"
                        f"Raw content preview: {item[:200]}...",
                        title="[error]JSON Parsing Error[/error]",
                        border_style="error",
                        box=box.ROUNDED,
                    ))
                else:
                    console.print(Panel(
                        f"[error]Item {i+1} is not a dictionary (got {type(item).__name__}).[/error]\n\n"
                        f"Expected format: [{{\"sequence\": \"...\"}}, ...] or [{{\"prompt\": \"...\"}}, ...]\n"
                        f"Got: {str(item)[:100]}",
                        title="[error]Invalid Input Format[/error]",
                        border_style="error",
                        box=box.ROUNDED,
                    ))
                sys.exit(1)
        
        console.print(f"[success]✓ Loaded {len(items)} item(s)[/success]")
        
        # Initialize model
        with console.status(f"[brand]Initializing {model_name} model...[/brand]"):
            model = Model(model_name)
        
        # When --batch-size is set: pre-make batches (list-of-lists) so client does not auto-batch
        if batch_size is not None:
            try:
                from biolm.core.http import BioLMApiClient
                import asyncio
                async def get_schema_max():
                    client = BioLMApiClient(model_name, raise_httpx=False)
                    try:
                        schema = await client.schema(model_name, action)
                        if schema:
                            return BioLMApiClient.extract_max_items(schema)
                    finally:
                        await client.shutdown()
                    return None
                schema_max = asyncio.run(get_schema_max())
                effective_batch_size = min(batch_size, schema_max) if schema_max else batch_size
            except Exception:
                effective_batch_size = batch_size
            payload = [items[i:i + effective_batch_size] for i in range(0, len(items), effective_batch_size)]
            if action == 'lookup':
                payload = items  # lookup API expects flat list of queries
        else:
            payload = items
        
        show_progress = (progress or len(items) > 1) and len(items) > 0
        total_items = len(items)
        
        if action == 'lookup':
            if show_progress:
                with console.status(f"[brand]Processing {total_items} item(s) with {model_name}...[/brand]"):
                    results = model.lookup(query=payload)
            else:
                results = model.lookup(query=payload)
            if not isinstance(results, builtins.list):
                results = [results]
        else:
            from biolm.progress import rich_progress
            if show_progress:
                with rich_progress(
                    total_items,
                    description=f"[brand]Processing {total_items} item(s) with {model_name}...[/brand]",
                    console=console,
                ) as progress_callback:
                    if action == 'encode':
                        results = model.encode(items=payload, params=params_dict, progress_callback=progress_callback)
                    elif action == 'predict':
                        results = model.predict(items=payload, params=params_dict, progress_callback=progress_callback)
                    elif action == 'generate':
                        results = model.generate(items=payload, params=params_dict, progress_callback=progress_callback)
                    else:
                        raise ValueError(f"Unknown action: {action}")
            else:
                with console.status(f"[brand]Processing with {model_name}...[/brand]"):
                    if action == 'encode':
                        results = model.encode(items=payload, params=params_dict)
                    elif action == 'predict':
                        results = model.predict(items=payload, params=params_dict)
                    elif action == 'generate':
                        results = model.generate(items=payload, params=params_dict)
                    else:
                        raise ValueError(f"Unknown action: {action}")
            if not isinstance(results, builtins.list):
                results = [results]
        
        # Save output
        # CRITICAL: --format is for OUTPUT format only
        # Priority: 1) --format option (explicit), 2) file extension, 3) default to JSON
        output_format = None
        
        # If --format is explicitly provided, use it (highest priority)
        if format:
            output_format = format
            console.print(f"[text.muted]Using output format from --format option: {output_format}[/text.muted]")
        elif output and output != '-':
            # Detect from output file extension if --format not specified
            detected_output_format = _detect_file_format(output)
            if detected_output_format != 'unknown':
                output_format = detected_output_format
                console.print(f"[text.muted]Output format detected from file extension: {output_format}[/text.muted]")
        
        # Default to JSON if still not set
        if not output_format:
            output_format = 'json'
            if output and output != '-':
                console.print(f"[text.muted]Defaulting to JSON output format[/text.muted]")
        
        with console.status(f"[brand]Saving results as {output_format}...[/brand]"):
            _save_output_data(results, output, output_format)
        
        if output and output != '-':
            console.print(f"[success]✓ Results saved to {output} ({len(results)} item(s))[/success]")
        elif not output:
            # If outputting to stdout and format is table, show summary
            if format and format != 'json':
                console.print(f"[success]✓ Processed {len(results)} item(s)[/success]")
    
    except FileNotFoundError as e:
        console.print(Panel(
            f"[error]File not found: {e}[/error]",
            title="[error]File Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    except ValueError as e:
        console.print(Panel(
            f"[error]{str(e)}[/error]",
            title="[error]Validation Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    except Exception as e:
        # Try to parse API error responses for better error messages
        error_msg = str(e)
        error_details = []
        error_dict = None
        
        # Check if it's an httpx exception with response body
        try:
            import httpx
            if isinstance(e, httpx.HTTPStatusError):
                # Try to get error from response
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        resp_json = e.response.json()
                        if isinstance(resp_json, dict):
                            error_dict = resp_json
                            error_msg = e.response.text
                        else:
                            error_msg = str(e.response.text)
                    except:
                        error_msg = str(e.response.text) if hasattr(e, 'response') else str(e)
        except ImportError:
            pass
        
        # Check if error is a dict/JSON string with API error format
        try:
            if isinstance(e, dict):
                error_dict = e
            elif not error_dict:
                # Try to parse error message as JSON (might be stringified)
                # Handle cases like: "{'error': {...}}" or '{"error": {...}}'
                cleaned_msg = error_msg.strip()
                # Replace single quotes with double quotes if needed
                if cleaned_msg.startswith("{'") or cleaned_msg.startswith("'{'"):
                    cleaned_msg = cleaned_msg.replace("'", '"')
                
                if cleaned_msg.startswith('{'):
                    error_dict = json.loads(cleaned_msg)
                elif hasattr(e, 'args') and e.args:
                    # Try parsing first argument if it's a string
                    first_arg = e.args[0]
                    if isinstance(first_arg, str):
                        cleaned_arg = first_arg.strip()
                        if cleaned_arg.startswith("{'") or cleaned_arg.startswith("'{'"):
                            cleaned_arg = cleaned_arg.replace("'", '"')
                        if cleaned_arg.startswith('{'):
                            error_dict = json.loads(cleaned_arg)
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass
        
        # Parse the error
        if error_dict and 'error' in error_dict:
            api_error = error_dict['error']
            
            # Parse field-level errors like "items__3__sequence"
            if isinstance(api_error, dict):
                for field_path, error_list in api_error.items():
                    # Extract item index from field path (e.g., "items__3__sequence" -> index 3)
                    if '__' in field_path:
                        parts = field_path.split('__')
                        if len(parts) >= 2 and parts[0] == 'items':
                            try:
                                item_index = int(parts[1])
                                field_name = '__'.join(parts[2:]) if len(parts) > 2 else 'unknown'
                                
                                # Get the problematic item
                                problematic_item = None
                                try:
                                    if items is not None and item_index < len(items):
                                        problematic_item = items[item_index]
                                except (IndexError, TypeError):
                                    pass
                                
                                error_details.append(f"[error]❌ Item {item_index + 1} ({field_name}):[/error]")
                                
                                # Show the problematic value
                                if problematic_item and field_name in problematic_item:
                                    value = problematic_item[field_name]
                                    if isinstance(value, str):
                                        # Show first and last 50 chars if too long
                                        if len(value) > 100:
                                            value_preview = value[:50] + '...' + value[-50:]
                                        else:
                                            value_preview = value
                                        # Highlight invalid characters
                                        invalid_chars = set(value) - set('ACDEFGHIKLMNPQRSTVWYBXZUO')
                                        if invalid_chars:
                                            error_details.append(f"  [text]Sequence contains invalid characters: {', '.join(sorted(invalid_chars))}[/text]")
                                        error_details.append(f"  [text]Sequence: {value_preview}[/text]")
                                    else:
                                        error_details.append(f"  [text]Value: {value}[/text]")
                                
                                # Show the error message
                                if isinstance(error_list, builtins.list) and error_list:
                                    error_details.append(f"  [error]Error: {error_list[0]}[/error]")
                                elif isinstance(error_list, str):
                                    error_details.append(f"  [error]Error: {error_list}[/error]")
                            except (ValueError, IndexError):
                                # Fall through to generic error handling
                                pass
            
            # If we couldn't parse it, show the raw error
            if not error_details:
                if isinstance(api_error, dict):
                    error_details.append(f"[error]{json.dumps(api_error, indent=2)}[/error]")
                else:
                    error_details.append(f"[error]{api_error}[/error]")
        else:
            error_details.append(f"[error]{error_msg}[/error]")
        
        # Build error message
        error_content = "\n".join(error_details) if error_details else f"[error]{error_msg}[/error]"
        
        # Add helpful context
        help_text = "\n\n[text.muted]💡 Tip: Check your input file for invalid characters in sequences.[/text.muted]"
        help_text += "\n[text.muted]Valid amino acid characters: ACDEFGHIKLMNPQRSTVWYBXZUO[/text.muted]"
        help_text += "\n[text.muted]Remove numbers, spaces, or other non-amino-acid characters from sequences.[/text.muted]"
        
        console.print(Panel(
            error_content + help_text,
            title="[error]❌ Error Running Model[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        
        if hasattr(e, '__cause__') and e.__cause__:
            console.print(f"[text.muted]Cause: {e.__cause__}[/text.muted]")
        sys.exit(1)


@model.command()
@click.argument('model_name', required=False)
@click.option('--action', '-a', help='Specific action (encode, predict, generate, lookup)')
@click.option('--format', '-f', type=click.Choice(['python', 'markdown', 'rst', 'json']), 
              default='python', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Output file path (default: stdout)')
def example(model_name, action, format, output):
    """Generate SDK usage examples for models.
    
    If model_name is not provided, lists all available models.
    """
    try:
        if model_name is None:
            # List all available models
            models = list_models(base_url=_client_catalog_base())
            if not models:
                console.print("[error]Could not fetch available models. Please check your connection and authentication.[/error]")
                return
            
            # Display models in a table
            table = Table(title="Available BioLM Models", show_header=True, header_style="brand.bold")
            table.add_column("Name", style="brand.bright")
            table.add_column("Slug", style="text.muted")
            table.add_column("Actions", style="text")
            
            for model in models[:50]:  # Limit to first 50 for display
                # Handle both old and new API response formats
                name = model.get('model_name') or model.get('name') or 'Unknown'
                slug = model.get('model_slug') or model.get('slug') or 'N/A'
                
                # Extract actions from boolean flags or actions array
                actions_list = []
                if 'actions' in model and isinstance(model['actions'], builtins.list):
                    actions_list = model['actions']
                else:
                    # Build actions list from boolean flags
                    if model.get('encoder'):
                        actions_list.append('encode')
                    if model.get('predictor'):
                        actions_list.append('predict')
                    if model.get('generator'):
                        actions_list.append('generate')
                    if model.get('classifier'):
                        actions_list.append('classify')
                    if model.get('similarity'):
                        actions_list.append('similarity')
                
                actions = ', '.join(actions_list) if actions_list else 'N/A'
                table.add_row(name, slug, actions)
            
            if len(models) > 50:
                table.add_row("...", f"({len(models) - 50} more models)", "")
            
            console.print(table)
            console.print(f"\n[text.muted]Use 'biolm model example <model_name>' to generate examples for a specific model.[/text.muted]")
        else:
            # Generate example for specific model
            with console.status("[brand]Generating example...[/brand]"):
                example_text = get_example(model_name, action=action, format=format)
            
            if output:
                # Write to file
                with open(output, 'w') as f:
                    f.write(example_text)
                console.print(f"[success]Example written to {output}[/success]")
            else:
                # Print to stdout
                console.print("\n[brand]SDK Usage Example[/brand]\n")
                console.print(Panel(
                    example_text,
                    title=f"[brand]{model_name}[/brand]",
                    border_style="brand",
                    box=box.ROUNDED,
                ))
    except Exception as e:
        console.print(f"[error]Error generating example: {e}[/error]")
        if hasattr(e, '__cause__') and e.__cause__:
            console.print(f"[text.muted]{e.__cause__}[/text.muted]")


@cli.group(cls=RichGroup)
def protocol():
    """Work with protocols.
    
    Commands for managing and executing BioLM protocols.
    """
    pass


@protocol.command()
def list():
    """List protocols.
    
    Display a list of all available protocols.
    """
    console.print(Panel(
        "[text.muted]Protocol commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to list and manage BioLM protocols.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@protocol.command()
@click.argument('protocol_source', required=False)
def show(protocol_source):
    """Show protocol details.
    
    Display a formatted report about a protocol configuration. The protocol can be
    specified either as a YAML file path or as a protocol ID from the platform.
    
    Examples:

    .. code-block:: bash

        # Show protocol from YAML file
        biolmai protocol show protocol.yaml

        # Show protocol from platform by ID
        biolmai protocol show abc123
    """
    from biolm.protocols import Protocol
    import os
    
    if not protocol_source:
        console.print(Panel(
            "[error]Protocol source required[/error]\n\n"
            "Specify either a YAML file path or a protocol ID from the platform.\n\n"
            "Examples:\n"
            "  biolm protocol show protocol.yaml\n"
            "  biolm protocol show abc123",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    try:
        # Check if it's a file path (exists on filesystem or has YAML extension)
        is_file = os.path.exists(protocol_source) or protocol_source.endswith(('.yaml', '.yml'))
        
        if is_file:
            # Treat as file path
            try:
                if not os.path.exists(protocol_source):
                    # File doesn't exist but has YAML extension - try anyway
                    console.print(Panel(
                        f"[warning]File not found: {protocol_source}[/warning]\n\n"
                        "Trying to load as YAML file...",
                        title="[warning]Warning[/warning]",
                        border_style="warning",
                        box=box.ROUNDED,
                    ))
                protocol_data = Protocol._load_yaml_static(protocol_source)
                Protocol.render_report(protocol_data, source=f"file: {protocol_source}", console=console)
            except FileNotFoundError:
                console.print(Panel(
                    f"[error]File not found: {protocol_source}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
            except ValueError as e:
                console.print(Panel(
                    f"[error]Invalid protocol data: {e}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
            except Exception as e:
                console.print(Panel(
                    f"[error]Failed to load protocol file: {e}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
        else:
            # Treat as protocol ID from platform
            try:
                with console.status("[brand]Fetching protocol from platform...[/brand]"):
                    protocol_data = Protocol.fetch_by_id(protocol_source)
                Protocol.render_report(protocol_data, source=f"platform: {protocol_source}", console=console)
            except FileNotFoundError as e:
                console.print(Panel(
                    f"[error]{str(e)}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
            except PermissionError as e:
                console.print(Panel(
                    f"[error]{str(e)}[/error]",
                    title="[error]Authentication Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
            except ValueError as e:
                # Could be from fetch_by_id or render_report
                console.print(Panel(
                    f"[error]{str(e)}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
            except Exception as e:
                console.print(Panel(
                    f"[error]Unexpected error: {e}[/error]",
                    title="[error]Error[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[text.muted]Cancelled.[/text.muted]")
        sys.exit(0)


@protocol.command()
@click.argument('protocol_file', type=click.Path(exists=True))
def run(protocol_file):
    """Run a protocol from a YAML file.
    
    Execute a protocol defined in a YAML file.
    """
    console.print(Panel(
        "[text.muted]Protocol commands are coming soon![/text.muted]\n\n"
        "This feature will allow you to execute protocols from YAML files.",
        title="[brand]Coming Soon[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))


@protocol.command()
@click.argument('protocol_file', type=click.Path(exists=True))
@click.option('--json', 'output_json', is_flag=True, help='Output results in JSON format')
def validate(protocol_file, output_json):
    """Validate a protocol YAML file.
    
    Check if a protocol YAML file is valid. Validates YAML syntax, JSON schema
    compliance, task references, circular dependencies, and template expressions.
    """
    from biolm.protocols import Protocol
    
    try:
        result = Protocol.validate(protocol_file)
    except Exception as e:
        if output_json:
            import json
            console.print(json.dumps({
                "valid": False,
                "errors": [{"message": str(e), "path": "", "error_type": "exception"}],
                "warnings": [],
                "statistics": {}
            }), markup=False, highlight=False)
        else:
            console.print(Panel(
                f"[error]Validation failed: {e}[/error]",
                title="[error]Error[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
        sys.exit(1)
    
    if output_json:
        import json
        output = {
            "valid": result.is_valid,
            "errors": [
                {
                    "message": err.message,
                    "path": err.path,
                    "error_type": err.error_type
                }
                for err in result.errors
            ],
            "warnings": result.warnings,
            "statistics": result.statistics
        }
        console.print(json.dumps(output, indent=2), markup=False, highlight=False)
        sys.exit(0 if result.is_valid else 1)
    
    # Rich formatted output
    if result.is_valid:
        # Success message with statistics
        stats = result.statistics
        stats_text = f"✓ Valid protocol"
        if stats:
            parts = []
            if "protocol_name" in stats:
                parts.append(f"'{stats['protocol_name']}'")
            if "task_count" in stats:
                parts.append(f"{stats['task_count']} task{'s' if stats['task_count'] != 1 else ''}")
            if "input_count" in stats:
                parts.append(f"{stats['input_count']} input{'s' if stats['input_count'] != 1 else ''}")
            if parts:
                stats_text += " with " + ", ".join(parts)
        
        console.print(Panel(
            stats_text,
            title="[success]✓ Validation Successful[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
        
        # Show warnings if any
        if result.warnings:
            console.print()
            for warning in result.warnings:
                console.print(f"[warning]⚠ {warning}[/warning]")
        
        # Show statistics table
        if stats and len(stats) > 1:  # More than just protocol_name
            console.print()
            table = Table(title="Protocol Statistics", show_header=True, header_style="brand.bold")
            table.add_column("Metric", style="text")
            table.add_column("Value", style="brand.bright")
            
            stat_labels = {
                "task_count": "Total Tasks",
                "model_task_count": "Model Tasks",
                "gather_task_count": "Gather Tasks",
                "input_count": "Inputs",
                "output_rule_count": "Output Rules"
            }
            
            for key, label in stat_labels.items():
                if key in stats:
                    table.add_row(label, str(stats[key]))
            
            if table.rows:
                console.print(table)
        
        sys.exit(0)
    else:
        # Error summary
        error_count = len(result.errors)
        console.print(Panel(
            f"[error]Validation failed with {error_count} error{'s' if error_count != 1 else ''}[/error]",
            title="[error]✗ Validation Failed[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        
        # Show warnings if any
        if result.warnings:
            console.print()
            for warning in result.warnings:
                console.print(f"[warning]⚠ {warning}[/warning]")
        
        # Show errors in a table
        if result.errors:
            console.print()
            error_table = Table(title="Validation Errors", show_header=True, header_style="error")
            error_table.add_column("#", style="text.muted", width=4)
            error_table.add_column("Type", style="error")
            error_table.add_column("Path", style="text.muted")
            error_table.add_column("Message", style="text")
            
            for i, err in enumerate(result.errors, 1):
                error_table.add_row(
                    str(i),
                    err.error_type,
                    err.path if err.path else "(root)",
                    err.message
                )
            
            console.print(error_table)
        
        sys.exit(1)


@protocol.command()
@click.argument('filename', required=False)
@click.option('--output', '-o', type=click.Path(), help='Output file path (default: protocol.yaml)')
@click.option('--example', '-e', help='Use an example template')
@click.option('--list-examples', is_flag=True, help='List available example templates')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing file')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode to select example')
def init(filename, output, example, list_examples, force, interactive):
    """Initialize a new protocol YAML file.
    
    Create a blank protocol YAML file or initialize from an example template.
    The generated file will be validated automatically.
    
    Examples:

    .. code-block:: bash

        # Create a blank protocol
        biolmai protocol init

        # Create with custom filename
        biolmai protocol init my_protocol.yaml

        # Create from example
        biolmai protocol init --example antibody_design

        # List available examples
        biolmai protocol init --list-examples

        # Interactive mode
        biolmai protocol init --interactive
    """
    from biolm.protocols import Protocol
    
    # List examples if requested
    if list_examples:
        examples = Protocol._list_available_examples()
        if not examples:
            console.print(Panel(
                "[text.muted]No example templates found in examples/ directory.[/text.muted]",
                title="[brand]Examples[/brand]",
                border_style="brand",
                box=box.ROUNDED,
            ))
            return
        
        table = Table(title="Available Protocol Examples", show_header=True, header_style="brand.bold")
        table.add_column("Name", style="brand.bright")
        table.add_column("File", style="text.muted")
        
        for ex in examples:
            table.add_row(ex, f"{ex}.yaml")
        
        console.print(table)
        console.print(f"\n[text.muted]Use 'biolm protocol init --example <name>' to create a protocol from an example.[/text.muted]")
        return
    
    # Determine output path
    if output:
        output_path = output
    elif filename:
        output_path = filename
    else:
        output_path = "protocol.yaml"
    
    # Interactive mode
    if interactive:
        examples = Protocol._list_available_examples()
        if not examples:
            console.print(Panel(
                "[error]No example templates available for interactive selection.[/error]",
                title="[error]Error[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Display examples in a table
        table = Table(title="Select an Example Template", show_header=True, header_style="brand.bold")
        table.add_column("#", style="text.muted", width=4)
        table.add_column("Name", style="brand.bright")
        table.add_column("File", style="text.muted")
        
        for i, ex in enumerate(examples, 1):
            table.add_row(str(i), ex, f"{ex}.yaml")
        
        console.print(table)
        console.print()
        
        # Prompt for selection
        try:
            choice = click.prompt(
                f"Select an example (1-{len(examples)})",
                type=click.IntRange(1, len(examples))
            )
            example = examples[choice - 1]
        except (click.Abort, KeyboardInterrupt):
            console.print("\n[text.muted]Cancelled.[/text.muted]")
            sys.exit(0)
    
    # Create the protocol file
    try:
        with console.status("[brand]Creating protocol file...[/brand]"):
            created_path = Protocol.init(output_path, example=example, force=force)
        
        # Display success message
        success_msg = f"[success]✓ Protocol file created successfully![/success]\n\n"
        success_msg += f"File: [brand]{created_path}[/brand]"
        
        if example:
            success_msg += f"\nTemplate: [text.muted]{example}[/text.muted]"
        
        success_msg += f"\n\n[text.muted]Use 'biolm protocol validate {created_path}' to validate the file.[/text.muted]"
        
        console.print(Panel(
            success_msg,
            title="[success]Success[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    
    except FileExistsError as e:
        console.print(Panel(
            f"[error]✗ {str(e)}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except FileNotFoundError as e:
        console.print(Panel(
            f"[error]✗ {str(e)}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except ValueError as e:
        console.print(Panel(
            f"[error]✗ {str(e)}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except Exception as e:
        console.print(Panel(
            f"[error]✗ Failed to create protocol file: {e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)


@protocol.command()
@click.argument('results', type=click.Path(exists=True))
@click.option('--outputs', type=click.Path(exists=True), help='Outputs config YAML or protocol YAML file')
@click.option('--account', required=True, help='Account name (experiment path: account/workspace/protocol)')
@click.option('--workspace', required=True, help='Workspace name (experiment path: account/workspace/protocol)')
@click.option('--protocol', 'protocol_slug', required=True, help='Protocol name/slug (experiment path: account/workspace/protocol)')
@click.option('--dry-run', is_flag=True, help='Prepare data without logging to MLflow')
@click.option('--mlflow-uri', default='https://mlflow.biolm.ai/', help='MLflow tracking URI')
@click.option('--aggregate-over', type=click.Choice(['selected', 'all']), default='selected',
              help='Compute aggregates over selected rows or all rows')
@click.option('--protocol-name', help='Protocol display name for metadata (default: from protocol YAML)')
@click.option('--protocol-version', help='Protocol version for metadata')
def log(results, outputs, account, workspace, protocol_slug, dry_run, mlflow_uri, aggregate_over, protocol_name, protocol_version):
    """Log protocol results to MLflow.
    
    Log protocol execution results to MLflow based on the protocol's outputs
    configuration. The MLflow experiment is created as account/workspace/protocol.
    
    Examples:

    .. code-block:: bash

        # Log results with outputs config from protocol file
        biolmai protocol log results.jsonl --outputs protocol.yaml --account acme --workspace lab --protocol antifold-antibody

        # Dry run to see what would be logged
        biolmai protocol log results.jsonl --outputs protocol.yaml --account acme --workspace lab --protocol antifold-antibody --dry-run

        # Use custom MLflow URI
        biolmai protocol log results.jsonl --outputs protocol.yaml --account acme --workspace lab --protocol antifold-antibody --mlflow-uri http://localhost:5001
    """
    try:
        from biolm.protocols_mlflow import (
            MLflowNotAvailableError,
            log_protocol_results,
        )
    except ImportError:
        console.print(Panel(
            "[error]MLflow logging functionality is not available.[/error]\n\n"
            "Install MLflow support with: [brand]pip install biolmai[mlflow][/brand]",
            title="[error]MLflow Not Available[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    if not outputs:
        console.print(Panel(
            "[error]--outputs option is required[/error]\n\n"
            "Specify the outputs configuration file (protocol YAML or outputs config YAML).",
            title="[error]Missing Outputs Config[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    try:
        # Prepare protocol metadata
        protocol_metadata = {}
        
        # Try to extract protocol name and version from protocol YAML if outputs is a protocol file
        if outputs and os.path.exists(outputs):
            try:
                import yaml
                with open(outputs, 'r') as f:
                    protocol_data = yaml.safe_load(f)
                    if isinstance(protocol_data, dict):
                        # Extract protocol name if not provided via CLI
                        if not protocol_name and "name" in protocol_data:
                            protocol_metadata["name"] = protocol_data["name"]
                        # Extract protocol version if not provided via CLI
                        if not protocol_version:
                            if "protocol_version" in protocol_data:
                                protocol_metadata["version"] = protocol_data["protocol_version"]
                            elif "about" in protocol_data and isinstance(protocol_data["about"], dict) and "version" in protocol_data["about"]:
                                protocol_metadata["version"] = protocol_data["about"]["version"]
                        # Extract inputs for parent run tags
                        if "inputs" in protocol_data and isinstance(protocol_data["inputs"], dict):
                            protocol_metadata["inputs"] = protocol_data["inputs"]
            except Exception:
                # If we can't load the protocol file, just continue with CLI-provided values
                pass
        
        # CLI-provided values override extracted values
        if protocol_name:
            protocol_metadata["name"] = protocol_name
        if protocol_version:
            protocol_metadata["version"] = protocol_version
        
        # Log results
        with console.status("[brand]Logging protocol results to MLflow...[/brand]"):
            result = log_protocol_results(
                results=results,
                outputs_config=outputs,
                account_name=account,
                workspace_name=workspace,
                protocol_name=protocol_slug,
                protocol_metadata=protocol_metadata if protocol_metadata else None,
                mlflow_uri=mlflow_uri,
                dry_run=dry_run,
                aggregate_over=aggregate_over,
            )
        
        # Display results
        if dry_run:
                # 1. Overall Summary Box
                summary_content = (
                    f"Experiment: [brand]{result['experiment_name']}[/brand]\n"
                    f"Results processed: [text]{result['num_results']}[/text]\n"
                    f"Results selected: [text]{result['num_selected']}[/text]\n"
                    f"Aggregates computed: [text]{result['num_aggregates']}[/text]\n\n"
                    f"[text.muted]No data was logged to MLflow (dry run mode).[/text.muted]"
                )
                console.print(Panel(
                    summary_content,
                    title="[success]Summary[/success]",
                    border_style="success",
                    box=box.ROUNDED,
                ))
                console.print()
                
                # 2. Protocol Run (Parent Run) Box with Tags, Parameters, and Aggregate Metrics
                if "prepared_data" in result:
                    prepared_data = result["prepared_data"]
                    parent_content_lines = []
                    
                    # Parent run tags
                    parent_tags = prepared_data.get("parent_tags", {})
                    parent_metadata = prepared_data.get("parent_metadata", {})
                    
                    # Combine parent_tags and parent_metadata (metadata becomes tags in MLflow)
                    all_parent_tags = {**parent_tags}
                    for key, value in parent_metadata.items():
                        if value is not None:
                            if key == "inputs" and isinstance(value, dict):
                                # Inputs are logged as individual tags with "input." prefix
                                for input_key, input_value in value.items():
                                    all_parent_tags[f"input.{input_key}"] = str(input_value)
                            else:
                                all_parent_tags[key] = str(value)
                    
                    if all_parent_tags:
                        parent_content_lines.append("[bold]Tags:[/bold]")
                        for tag_name, tag_value in sorted(all_parent_tags.items()):
                            parent_content_lines.append(f"  {tag_name}: [brand]{tag_value}[/brand]")
                        parent_content_lines.append("")
                    
                    # Parent run parameters (currently none, but structure is here)
                    parent_params = prepared_data.get("parent_params", {})
                    if parent_params:
                        parent_content_lines.append("[bold]Parameters:[/bold]")
                        for param_name, param_value in sorted(parent_params.items()):
                            parent_content_lines.append(f"  {param_name}: [brand]{param_value}[/brand]")
                        parent_content_lines.append("")
                    
                    # Aggregate metrics
                    aggregate_metrics = prepared_data.get("aggregate_metrics", {})
                    if aggregate_metrics:
                        parent_content_lines.append("[bold]Aggregate Metrics:[/bold]")
                        for metric_name, metric_value in sorted(aggregate_metrics.items()):
                            if isinstance(metric_value, float):
                                parent_content_lines.append(f"  {metric_name}: [brand]{metric_value:.6f}[/brand]")
                            else:
                                parent_content_lines.append(f"  {metric_name}: [brand]{metric_value}[/brand]")
                    
                    if parent_content_lines:
                        console.print(Panel(
                            "\n".join(parent_content_lines),
                            title="[brand]Protocol Run (Parent Run)[/brand]",
                            border_style="brand",
                            box=box.ROUNDED,
                        ))
                        console.print()
                
                # 3. Table of Selected Results with MLflow Logging Fields
                if "prepared_data" in result and result["prepared_data"].get("child_runs"):
                    try:
                        child_runs_list = result["prepared_data"]["child_runs"]
                        
                        # Create main table for selected results
                        results_table = Table(
                            title="Selected Results (Output Records)",
                            show_header=True,
                            header_style="brand.bold",
                            box=box.ROUNDED,
                            title_style="brand.bright",
                        )
                        results_table.add_column("#", style="text.muted", width=4, justify="right")
                        results_table.add_column("Parameters", style="text", width=25)
                        results_table.add_column("Metrics", style="text", width=25)
                        results_table.add_column("Tags", style="text", width=20)
                        results_table.add_column("Artifacts", style="text", width=20)
                        
                        for idx, child_data in enumerate(child_runs_list, 1):
                            # Format parameters
                            params = child_data.get("params", {})
                            params_str = ", ".join([f"{k}={v}" for k, v in params.items()]) if params else "[text.muted]—[/text.muted]"
                            if len(params_str) > 100:
                                params_str = params_str[:97] + "..."
                            
                            # Format metrics
                            metrics = child_data.get("metrics", {})
                            metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]) if metrics else "[text.muted]—[/text.muted]"
                            if len(metrics_str) > 100:
                                metrics_str = metrics_str[:97] + "..."
                            
                            # Format tags (include automatically added "type": "model")
                            tags = child_data.get("tags", {}).copy()
                            tags["type"] = "model"  # This is automatically added in MLflow logging
                            tags_str = ", ".join([f"{k}={v}" for k, v in sorted(tags.items())]) if tags else "[text.muted]—[/text.muted]"
                            if len(tags_str) > 80:
                                tags_str = tags_str[:77] + "..."
                            
                            # Format artifacts
                            artifacts = child_data.get("artifacts", [])
                            if artifacts:
                                artifact_list = []
                                for artifact_name, artifact_content in artifacts:
                                    size = len(artifact_content) if isinstance(artifact_content, str) else len(str(artifact_content))
                                    artifact_list.append(f"{artifact_name} ({size} bytes)")
                                artifacts_str = ", ".join(artifact_list)
                            else:
                                artifacts_str = "[text.muted]—[/text.muted]"
                            if len(artifacts_str) > 80:
                                artifacts_str = artifacts_str[:77] + "..."
                            
                            results_table.add_row(
                                str(idx),
                                params_str,
                                metrics_str,
                                tags_str,
                                artifacts_str
                            )
                        
                        console.print(results_table)
                        console.print()
                    except Exception as e:
                        console.print(f"[error]Error displaying selected results: {e}[/error]")
                        import traceback
                        import sys
                        exc_type, exc_value, exc_tb = sys.exc_info()
                        console.print(f"[text.muted]Exception type: {exc_type.__name__}[/text.muted]")
                        console.print(f"[text.muted]Exception message: {str(exc_value)}[/text.muted]")
        else:
            console.print(Panel(
                f"[success]✓ Results logged successfully![/success]\n\n"
                f"Experiment: [brand]{result['experiment_name']}[/brand]\n"
                f"Parent run ID: [text]{result['parent_run_id']}[/text]\n"
                f"Child runs: [text]{len(result['child_run_ids'])}[/text]\n"
                f"Results processed: [text]{result['num_results']}[/text]\n"
                f"Results selected: [text]{result['num_selected']}[/text]\n"
                f"Aggregates computed: [text]{result['num_aggregates']}[/text]",
                title="[success]Logging Complete[/success]",
                border_style="success",
                box=box.ROUNDED,
            ))
    
    except MLflowNotAvailableError as e:
        console.print(Panel(
            f"[error]{str(e)}[/error]",
            title="[error]MLflow Not Available[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except FileNotFoundError as e:
        console.print(Panel(
            f"[error]File not found: {str(e)}[/error]",
            title="[error]File Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except ValueError as e:
        console.print(Panel(
            f"[error]{str(e)}[/error]",
            title="[error]Validation Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    
    except Exception as e:
        console.print(Panel(
            f"[error]Unexpected error: {str(e)}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)


@cli.group(cls=RichGroup)
def dataset():
    """Manage datasets.
    
    Commands for creating, listing, and managing datasets.
    """
    pass


@dataset.command()
@click.option('--experiment', default=None, help='MLflow experiment name (default: {username}/datasets)')
@click.option('--format', type=click.Choice(['table', 'json', 'csv']), default='table', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Save output to file')
@click.option('--mlflow-uri', default='https://mlflow.biolm.ai/', help='MLflow tracking URI')
@click.option('--all-runs', is_flag=True, help='List all runs in experiment, not just datasets (for debugging)')
def list(experiment, format, output, mlflow_uri, all_runs):
    """List datasets.
    
    Display a list of all datasets you have access to.
    
    Examples:

    .. code-block:: bash

        # List all datasets
        biolmai dataset list

        # List datasets in specific experiment
        biolmai dataset list --experiment my-datasets

        # Output as JSON
        biolmai dataset list --format json --output datasets.json

        # List all runs (for debugging - shows runs without dataset tag)
        biolmai dataset list --all-runs
    """
    try:
        # Check MLflow availability
        try:
            from biolm.datasets_mlflow import _check_mlflow_available
            _check_mlflow_available()
        except MLflowNotAvailableError:
            console.print(Panel(
                "[error]MLflow logging functionality is not available.[/error]\n\n"
                "Install MLflow support with: [brand]pip install biolmai[mlflow][/brand]",
                title="[error]MLflow Not Available[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Check credentials (consistent with login/status)
        if not are_credentials_valid():
            console.print(Panel(
                "[error]Authentication required.[/error]\n\n"
                "Please run [brand]biolm login[/brand] to authenticate.",
                title="[error]Not Authenticated[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        with console.status("[brand]Fetching datasets...[/brand]"):
            try:
                datasets = list_datasets(experiment_name=experiment, mlflow_uri=mlflow_uri, all_runs=all_runs)
            except RuntimeError as e:
                console.print(Panel(
                    f"[error]{str(e)}[/error]\n\n"
                    "This might indicate:\n"
                    "- Authentication issues with MLflow\n"
                    "- Network connectivity problems\n"
                    "- MLflow server configuration issues",
                    title="[error]Error Fetching Datasets[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
        
        if not datasets:
            console.print(Panel(
                "[text.muted]No datasets found in experiment '{experiment}'.[/text.muted]\n\n"
                "Datasets are created automatically when you upload files.\n"
                "Try: [brand]biolm dataset upload my-dataset-123 data.csv[/brand]".format(experiment=experiment),
                title="[text.muted]No Datasets[/text.muted]",
                border_style="border",
                box=box.ROUNDED,
            ))
            sys.exit(0)
        
        # Output based on format
        if format == 'json':
            output_data = json.dumps(datasets, indent=2, default=str)
            if output:
                with open(output, 'w') as f:
                    f.write(output_data)
                console.print(f"[success]✓ Datasets saved to {output}[/success]")
            else:
                console.print(output_data, markup=False, highlight=False)
        elif format == 'csv':
            if not datasets:
                console.print("[text.muted]No datasets to display.[/text.muted]")
                sys.exit(0)
            
            # Flatten datasets for CSV
            csv_data = []
            for dataset in datasets:
                row = {
                    'dataset_id': dataset.get('dataset_id', ''),
                    'name': dataset.get('name', ''),
                    'run_id': dataset.get('run_id', ''),
                    'status': dataset.get('status', ''),
                    'artifact_count': dataset.get('artifact_count', 0),
                    'start_time': dataset.get('start_time', ''),
                }
                csv_data.append(row)
            
            to_csv(csv_data, output if output else '-')
            if output:
                console.print(f"[success]✓ Datasets saved to {output}[/success]")
        else:  # table format
            table = Table(
                title="[brand]📊 Available Datasets[/brand]",
                show_header=True,
                header_style="brand.bold",
                box=box.ROUNDED,
                title_style="brand.bright",
            )
            
            table.add_column("Dataset ID", style="brand", width=20)
            table.add_column("Name", style="text", width=25)
            table.add_column("Status", style="text", width=10)
            table.add_column("Artifacts", style="text", width=10)
            table.add_column("Run ID", style="text.muted", width=20)
            
            for dataset in datasets:
                dataset_id = dataset.get('dataset_id', 'N/A')
                name = dataset.get('name', 'N/A')
                status = dataset.get('status', 'UNKNOWN')
                artifact_count = dataset.get('artifact_count', 0)
                run_id = dataset.get('run_id', 'N/A')
                
                # Format status with color
                if status == 'FINISHED':
                    status_str = f"[success]{status}[/success]"
                elif status == 'RUNNING':
                    status_str = f"[brand]{status}[/brand]"
                elif status == 'FAILED':
                    status_str = f"[error]{status}[/error]"
                else:
                    status_str = status
                
                table.add_row(
                    dataset_id,
                    name,
                    status_str,
                    str(artifact_count),
                    run_id[:18] + "..." if len(run_id) > 18 else run_id,
                )
            
            console.print(table)
            if output:
                # Also save table data to file as JSON
                with open(output, 'w') as f:
                    json.dump(datasets, f, indent=2, default=str)
                console.print(f"\n[success]✓ Dataset data saved to {output}[/success]")
    
    except Exception as e:
        console.print(Panel(
            f"[error]Error listing datasets: {e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        if hasattr(e, '__cause__') and e.__cause__:
            console.print(f"[text.muted]{e.__cause__}[/text.muted]")
        sys.exit(1)


@dataset.command()
@click.argument('dataset_id', required=True)
@click.option('--experiment', default=None, help='MLflow experiment name (default: {username}/datasets)')
@click.option('--format', type=click.Choice(['table', 'json', 'yaml']), default='table', help='Output format')
@click.option('--output', '-o', type=click.Path(), help='Save output to file')
@click.option('--mlflow-uri', default='https://mlflow.biolm.ai/', help='MLflow tracking URI')
def show(dataset_id, experiment, format, output, mlflow_uri):
    """Show dataset details.
    
    Display detailed information about a specific dataset, including metadata,
    tags, parameters, metrics, and artifacts.
    
    By default, looks for datasets in the ``{username}/datasets`` experiment.
    
    Examples:

    .. code-block:: bash

        # Show dataset details
        biolmai dataset show my-dataset-123

        # Output as JSON
        biolmai dataset show my-dataset-123 --format json --output dataset.json
    """
    try:
        # Check MLflow availability
        try:
            from biolm.datasets_mlflow import _check_mlflow_available
            _check_mlflow_available()
        except MLflowNotAvailableError:
            console.print(Panel(
                "[error]MLflow logging functionality is not available.[/error]\n\n"
                "Install MLflow support with: [brand]pip install biolmai[mlflow][/brand]",
                title="[error]MLflow Not Available[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Check credentials (consistent with login/status)
        if not are_credentials_valid():
            console.print(Panel(
                "[error]Authentication required.[/error]\n\n"
                "Please run [brand]biolm login[/brand] to authenticate.",
                title="[error]Not Authenticated[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        with console.status(f"[brand]Fetching dataset '{dataset_id}'...[/brand]"):
            dataset = get_dataset(dataset_id, experiment_name=experiment, mlflow_uri=mlflow_uri)
        
        if not dataset:
            console.print(Panel(
                f"[error]Dataset '{dataset_id}' not found.[/error]\n\n"
                f"Use 'biolm dataset list' to see available datasets.",
                title="[error]Dataset Not Found[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Output based on format
        if format == 'json':
            output_data = json.dumps(dataset, indent=2, default=str)
            if output:
                with open(output, 'w') as f:
                    f.write(output_data)
                console.print(f"[success]✓ Dataset details saved to {output}[/success]")
            else:
                console.print(output_data, markup=False, highlight=False)
        elif format == 'yaml':
            try:
                import yaml
                output_data = yaml.dump(dataset, default_flow_style=False, allow_unicode=True)
                if output:
                    with open(output, 'w') as f:
                        f.write(output_data)
                    console.print(f"[success]✓ Dataset details saved to {output}[/success]")
                else:
                    console.print(output_data)
            except ImportError:
                console.print(Panel(
                    "[error]YAML support not available.[/error]\n\n"
                    "Install with: pip install pyyaml",
                    title="[error]Missing Dependency[/error]",
                    border_style="error",
                    box=box.ROUNDED,
                ))
                sys.exit(1)
        else:  # table format
            # Create main info table
            info_table = Table(
                title=f"[brand]📊 Dataset: {dataset.get('name', dataset_id)}[/brand]",
                show_header=False,
                box=box.ROUNDED,
                title_style="brand.bright",
            )
            
            info_table.add_column("Field", style="brand.bold", width=20)
            info_table.add_column("Value", style="text", width=50)
            
            info_table.add_row("Dataset ID", dataset.get('dataset_id', 'N/A'))
            info_table.add_row("Run ID", dataset.get('run_id', 'N/A'))
            info_table.add_row("Name", dataset.get('name', 'N/A'))
            info_table.add_row("Status", dataset.get('status', 'UNKNOWN'))
            
            # Format timestamps
            start_time = dataset.get('start_time')
            if start_time:
                from datetime import datetime
                try:
                    dt = datetime.fromtimestamp(start_time / 1000.0)
                    info_table.add_row("Start Time", dt.strftime("%Y-%m-%d %H:%M:%S"))
                except:
                    info_table.add_row("Start Time", str(start_time))
            
            console.print(info_table)
            console.print()
            
            # Tags table
            tags = dataset.get('tags', {})
            if tags:
                tags_table = Table(
                    title="[brand]Tags[/brand]",
                    show_header=True,
                    header_style="brand.bold",
                    box=box.ROUNDED,
                )
                tags_table.add_column("Key", style="brand")
                tags_table.add_column("Value", style="text")
                
                for key, value in sorted(tags.items()):
                    tags_table.add_row(key, str(value))
                
                console.print(tags_table)
                console.print()
            
            # Parameters table
            params = dataset.get('params', {})
            if params:
                params_table = Table(
                    title="[brand]Parameters[/brand]",
                    show_header=True,
                    header_style="brand.bold",
                    box=box.ROUNDED,
                )
                params_table.add_column("Key", style="brand")
                params_table.add_column("Value", style="text")
                
                for key, value in sorted(params.items()):
                    params_table.add_row(key, str(value))
                
                console.print(params_table)
                console.print()
            
            # Metrics table
            metrics = dataset.get('metrics', {})
            if metrics:
                metrics_table = Table(
                    title="[brand]Metrics[/brand]",
                    show_header=True,
                    header_style="brand.bold",
                    box=box.ROUNDED,
                )
                metrics_table.add_column("Key", style="brand")
                metrics_table.add_column("Value", style="text")
                
                for key, value in sorted(metrics.items()):
                    metrics_table.add_row(key, str(value))
                
                console.print(metrics_table)
                console.print()
            
            # Artifacts table
            artifacts = dataset.get('artifacts', [])
            if artifacts:
                artifacts_table = Table(
                    title="[brand]Artifacts[/brand]",
                    show_header=True,
                    header_style="brand.bold",
                    box=box.ROUNDED,
                )
                artifacts_table.add_column("Path", style="brand")
                artifacts_table.add_column("Type", style="text")
                artifacts_table.add_column("Size", style="text.muted")
                
                for artifact in artifacts:
                    path = artifact.get('path', 'N/A')
                    is_dir = artifact.get('is_dir', False)
                    file_size = artifact.get('file_size')
                    size_str = f"{file_size} bytes" if file_size else "N/A"
                    type_str = "Directory" if is_dir else "File"
                    artifacts_table.add_row(path, type_str, size_str)
                
                console.print(artifacts_table)
            else:
                console.print("[text.muted]No artifacts found.[/text.muted]")
            
            if output:
                # Also save dataset data to file as JSON
                with open(output, 'w') as f:
                    json.dump(dataset, f, indent=2, default=str)
                console.print(f"\n[success]✓ Dataset data saved to {output}[/success]")
    
    except Exception as e:
        console.print(Panel(
            f"[error]Error showing dataset: {e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        if hasattr(e, '__cause__') and e.__cause__:
            console.print(f"[text.muted]{e.__cause__}[/text.muted]")
        sys.exit(1)


@dataset.command()
@click.argument('dataset_id')
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--experiment', default=None, help='MLflow experiment name (default: {username}/datasets)')
@click.option('--name', help='Dataset name/description (stored as run name)')
@click.option('--recursive', '-r', is_flag=True, help='Upload directory recursively')
@click.option('--mlflow-uri', default='https://mlflow.biolm.ai/', help='MLflow tracking URI')
def upload(dataset_id, file_path, experiment, name, recursive, mlflow_uri):
    """Upload data to a dataset.
    
    Upload data from a file or directory to the specified dataset.
    If the dataset doesn't exist, it will be created automatically.
    
    Examples:

    .. code-block:: bash

        # Upload a single file
        biolmai dataset upload my-dataset-123 data.csv

        # Upload a directory
        biolmai dataset upload my-dataset-123 ./data --recursive

        # Upload with a custom name
        biolmai dataset upload my-dataset-123 data.csv --name "Training Data"
    """
    try:
        # Check MLflow availability
        try:
            from biolm.datasets_mlflow import _check_mlflow_available
            _check_mlflow_available()
        except MLflowNotAvailableError:
            console.print(Panel(
                "[error]MLflow logging functionality is not available.[/error]\n\n"
                "Install MLflow support with: [brand]pip install biolmai[mlflow][/brand]",
                title="[error]MLflow Not Available[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Check credentials (consistent with login/status)
        if not are_credentials_valid():
            console.print(Panel(
                "[error]Authentication required.[/error]\n\n"
                "Please run [brand]biolm login[/brand] to authenticate.",
                title="[error]Not Authenticated[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            console.print(Panel(
                f"[error]File or directory not found: {file_path}[/error]",
                title="[error]File Not Found[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        with console.status(f"[brand]Uploading to dataset '{dataset_id}'...[/brand]"):
            result = upload_dataset(
                dataset_id=dataset_id,
                file_path=file_path,
                experiment_name=experiment,
                name=name,
                mlflow_uri=mlflow_uri,
                recursive=recursive
            )
        
        console.print(Panel(
            f"[success]✓ Successfully uploaded to dataset '{dataset_id}'[/success]\n\n"
            f"Run ID: {result.get('run_id', 'N/A')}\n"
            f"Dataset ID: {result.get('dataset_id', 'N/A')}",
            title="[success]Upload Complete[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    
    except FileNotFoundError as e:
        console.print(Panel(
            f"[error]{str(e)}[/error]",
            title="[error]File Not Found[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            f"[error]Error uploading dataset: {e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        if hasattr(e, '__cause__') and e.__cause__:
            console.print(f"[text.muted]{e.__cause__}[/text.muted]")
        sys.exit(1)


@dataset.command()
@click.argument('dataset_id')
@click.argument('output_path', type=click.Path(), required=False, default='.')
@click.option('--experiment', default=None, help='MLflow experiment name (default: {username}/datasets)')
@click.option('--artifact-path', help='Specific artifact path to download (default: all artifacts)')
@click.option('--mlflow-uri', default='https://mlflow.biolm.ai/', help='MLflow tracking URI')
def download(dataset_id, output_path, experiment, artifact_path, mlflow_uri):
    """Download a dataset.
    
    Download all artifacts from a dataset to the specified directory.
    
    By default, looks for datasets in the ``{username}/datasets`` experiment.
    
    Examples:

    .. code-block:: bash

        # Download all artifacts to current directory
        biolmai dataset download my-dataset-123

        # Download to specific directory
        biolmai dataset download my-dataset-123 ./downloads

        # Download specific artifact
        biolmai dataset download my-dataset-123 ./downloads --artifact-path model.pkl
    """
    try:
        # Check MLflow availability
        try:
            from biolm.datasets_mlflow import _check_mlflow_available
            _check_mlflow_available()
        except MLflowNotAvailableError:
            console.print(Panel(
                "[error]MLflow logging functionality is not available.[/error]\n\n"
                "Install MLflow support with: [brand]pip install biolmai[mlflow][/brand]",
                title="[error]MLflow Not Available[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        # Check credentials (consistent with login/status)
        if not are_credentials_valid():
            console.print(Panel(
                "[error]Authentication required.[/error]\n\n"
                "Please run [brand]biolm login[/brand] to authenticate.",
                title="[error]Not Authenticated[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)
        
        output_path_obj = Path(output_path)
        
        with console.status(f"[brand]Downloading dataset '{dataset_id}'...[/brand]"):
            result = download_dataset(
                dataset_id=dataset_id,
                output_path=output_path,
                experiment_name=experiment,
                artifact_path=artifact_path,
                mlflow_uri=mlflow_uri
            )
        
        console.print(Panel(
            f"[success]✓ Successfully downloaded dataset '{dataset_id}'[/success]\n\n"
            f"Download location: {result.get('output_path', output_path)}\n"
            f"Run ID: {result.get('run_id', 'N/A')}",
            title="[success]Download Complete[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    
    except ValueError as e:
        console.print(Panel(
            f"[error]{str(e)}[/error]",
            title="[error]Dataset Not Found[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            f"[error]Error downloading dataset: {e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        if hasattr(e, '__cause__') and e.__cause__:
            console.print(f"[text.muted]{e.__cause__}[/text.muted]")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
