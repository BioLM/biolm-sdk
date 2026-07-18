"""Console script for biolm."""
from __future__ import annotations

import copy
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Union, Any, List, Dict
import builtins

import click
from click.formatting import HelpFormatter
from biolm import __version__ as BIOLM_VERSION
from biolm.cli.theme import create_console, no_color_requested
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
    BIOLM_PUBLIC_CLIENT_ID,
    get_env_api_token,
    get_model_catalog_base,
    get_model_api_source,
    is_hub_mode,
)
from biolm.models.examples import get_example, list_models, get_model_details
from biolm.io import load_fasta, load_csv, load_pdb, load_json, to_fasta, to_csv, to_pdb, to_json
from biolm.models import Model
from biolm.platform import PlatformClient, PlatformError, Workspace
from biolm.protocol_runs import ProtocolClient, ProtocolRunError

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


def _command_help_line(cmd: click.Command) -> str:
    """Return the first docstring line for help listings (not Click's truncated short help)."""
    if cmd.help:
        for line in cmd.help.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return cmd.get_short_help_str(limit=120) or ""


def _iter_visible_leaf_paths(
    group: click.Group,
    prefix: tuple[str, ...] = (),
) -> builtins.list[tuple[str, click.Command]]:
    """Return ``(path, command)`` for every visible leaf under *group*.

    Paths are space-joined (e.g. ``dataset create``). Hidden commands and
    hidden aliases are skipped. Nested groups are walked but not listed as rows.
    """
    rows: builtins.list[tuple[str, click.Command]] = []
    for name, cmd in sorted(group.commands.items(), key=lambda item: item[0]):
        if cmd.hidden:
            continue
        path_parts = prefix + (name,)
        if isinstance(cmd, click.Group):
            rows.extend(_iter_visible_leaf_paths(cmd, path_parts))
        else:
            rows.append((" ".join(path_parts), cmd))
    return rows


def _root_help_sections(
    group: click.Group,
) -> builtins.list[tuple[str, builtins.list[tuple[str, click.Command]]]]:
    """Group visible root leaves into panels keyed by top-level command group.

    Standalone root leaves (e.g. ``status``, ``whoami``) share an Account panel
    with the ``account`` group. Other top-level groups get their own panel titled
    with a pluralized label when useful.
    """
    section_order = (
        "Account",
        "Workspace",
        "Hub",
        "Models",
        "Protocols",
        "Datasets",
        "Commands",
    )
    title_for_root = {
        "account": "Account",
        "status": "Account",
        "whoami": "Account",
        "workspace": "Workspace",
        "hub": "Hub",
        "model": "Models",
        "protocol": "Protocols",
        "dataset": "Datasets",
    }
    sections: dict[str, builtins.list[tuple[str, click.Command]]] = {
        title: [] for title in section_order
    }

    for name, cmd in sorted(group.commands.items(), key=lambda item: item[0]):
        if cmd.hidden:
            continue
        title = title_for_root.get(name, "Commands")
        if isinstance(cmd, click.Group):
            sections[title].extend(_iter_visible_leaf_paths(cmd, (name,)))
        else:
            sections[title].append((name, cmd))

    return [(title, rows) for title, rows in sections.items() if rows]


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
        
        # Root help: exhaustive leaf paths, one panel per top-level group.
        # Nested group help: direct visible children only (relative names).
        if ctx.parent is None:
            sections = _root_help_sections(self)
        else:
            sections = [
                (
                    "Commands",
                    [
                        (name, cmd)
                        for name, cmd in sorted(
                            self.commands.items(), key=lambda item: item[0]
                        )
                        if not cmd.hidden
                    ],
                )
            ]

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
        
        for section_title, entries in sections:
            if not entries:
                continue
            name_width = max(25, max(len(path) for path, _ in entries))
            box_content = []
            for name, cmd in entries:
                help_text = _command_help_line(cmd)
                cmd_padding = " " * max(0, name_width - len(name))
                line = (
                    f"[brand.bright]{name}[/brand.bright]{cmd_padding}  "
                    f"[text]{help_text}[/text]"
                )
                box_content.append(line)

            panel = Panel(
                "\n".join(box_content),
                title=f"[bold]{section_title}[/bold]",
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


def _hidden_leaf_alias(parent, name, target):
    """Register a hidden copy of a leaf command under ``parent``."""
    if isinstance(target, click.Group):
        raise TypeError("hidden aliases must target leaf commands")
    alias = copy.copy(target)
    alias.name = name
    alias.hidden = True
    parent.add_command(alias, name)
    return alias


@click.command()
def main(args=None):
    """Console script for biolm."""
    click.echo("Replace this message by putting your code into " "biolm.cli")
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
@click.pass_context
def cli(ctx, debug, color):
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

    if ctx.invoked_subcommand is None:
        ctx.command.format_help(ctx, click.HelpFormatter())


@cli.command(hidden=True)
def version():
    """Print the installed ``biolm`` package version."""
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
    table.add_column("Value")

    api_token = get_env_api_token()
    if api_token:
        masked = f"{api_token[:6]}••••••••" if len(api_token) >= 6 else "••••••••"
        if os.environ.get("BIOLM_TOKEN"):
            table.add_row("BIOLM_TOKEN", masked)
        else:
            table.add_row("BIOLMAI_TOKEN", f"{masked} [text.muted](deprecated)[/text.muted]")
    else:
        table.add_row("BIOLM_TOKEN", "[text.muted]Not set[/text.muted]")

    table.add_row("Credentials Path", str(ACCESS_TOK_PATH))
    table.add_row("Model API URL", BIOLM_BASE_API_URL)
    table.add_row("Model API source", get_model_api_source())
    table.add_row("Platform Domain", BIOLM_BASE_DOMAIN)
    if is_hub_mode():
        table.add_row("Hub mode", "[success]yes[/success]")
    catalog_base = _client_catalog_base()
    if catalog_base.rstrip("/") != BIOLM_BASE_DOMAIN.rstrip("/"):
        table.add_row("Model Catalog Host", catalog_base)

    console.print(table)


def _display_status_context() -> None:
    """Best-effort platform context that never makes status fail."""
    unavailable = "[text.muted]unavailable[/text.muted]"
    account_value = unavailable
    workspace_value = unavailable
    client = None
    try:
        client = PlatformClient()
        try:
            active = client.current_workspace()
        except PlatformError:
            pass
        else:
            account_value = "{} {} ({})".format(
                active.account_type,
                active.account,
                active.account_id,
            )
            workspace_value = active.path
    except PlatformError:
        pass
    finally:
        if client is not None:
            client.close()

    table = Table(
        title="[brand]Active Platform Context[/brand]",
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bright",
    )
    table.add_column("Setting", style="brand", no_wrap=True)
    table.add_column("Value")
    table.add_row("Account", account_value)
    table.add_row("Workspace", workspace_value)
    console.print(table)


@cli.command()
def status():
    """Show authentication status, API endpoints, and where credentials are stored.

    Prints environment variables, the active model API URL, hub mode, and validates
    saved OAuth credentials when present.
    """
    display_env_vars_table()
    console.print()  # Add spacing before auth status
    get_auth_status()
    console.print()
    _display_status_context()


@cli.group(cls=RichGroup)
def account():
    """Manage BioLM account authentication, usage, budget, API keys, and organizations."""
    pass


@account.command()
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
    """Log in to BioLM with OAuth 2.0 (PKCE) and save credentials to ``~/.biolm/credentials``.

    Reuses valid existing credentials when possible; otherwise opens a browser to complete authorization.
    
    Examples:

    .. code-block:: bash

        # Login with default client ID
        biolm account login

        # Login with custom client ID
        biolm account login --client-id your-client-id

        # Login with custom scope (supported: read, write, introspection)
        biolm account login --scope "read write"
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


@account.command()
def logout():
    """Log out and remove saved OAuth credentials from ``~/.biolm/credentials``.

    After logout you must run ``biolm account login`` again before calling authenticated commands.
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


_hidden_leaf_alias(cli, "login", login)
_hidden_leaf_alias(cli, "logout", logout)


@cli.group(cls=RichGroup)
def hub():
    """Route model inference through a local or self-hosted biolm-hub gateway.

    Use ``biolm hub set`` after ``bh serve`` (or pointing at a deployed gateway). Platform
    login and protocol commands still target biolm.ai unless overridden in the environment.
    """
    pass


@hub.command("set")
@click.argument("url", required=False, default="http://127.0.0.1:8000")
def hub_set(url):
    """Save a biolm-hub gateway URL so ``biolm model`` uses local or self-hosted inference.

    Writes ``hub_api_url`` to ``~/.biolm/config.yaml``. Platform login and protocols still
    use biolm.ai unless ``BIOLM_BASE_DOMAIN`` is set.
    """
    from biolm.hub.config import hub_origin, normalize_hub_url, write_hub_api_url
    from biolm.hub.discovery import fetch_hub_status

    api_url = normalize_hub_url(url)
    with console.status(f"[brand]Connecting to {api_url}...[/brand]"):
        status = fetch_hub_status(api_url)

    if not status.get("healthy"):
        console.print(Panel(
            f"[error]Could not reach biolm-hub at {api_url}.[/error]\n\n"
            f"{status.get('message') or 'Start bh serve in the biolm-hub repo, then retry.'}",
            title="[error]Hub Unavailable[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        raise click.Abort()

    config_file = write_hub_api_url(api_url)
    origin = hub_origin(api_url)
    console.print(Panel(
        f"[success]Connected to biolm-hub[/success]\n\n"
        f"API: [brand]{api_url}[/brand]\n"
        f"Catalog: [brand]{origin}/catalog[/brand]\n"
        f"Variants: {status.get('slug_count', 0)} "
        f"({status.get('route_count', 0)} routes)\n"
        f"Config: {config_file}\n\n"
        f"Run [brand]biolm model list[/brand] or "
        f"[brand]biolm model run esm2-8m encode -i seq.json[/brand]",
        title="[brand]biolm hub[/brand]",
        border_style="success",
        box=box.ROUNDED,
    ))


@hub.command("status")
def hub_status():
    """Show saved hub configuration, the active model API URL, and live gateway health."""
    from biolm.hub.config import config_path, hub_origin, read_hub_api_url
    from biolm.hub.discovery import fetch_hub_status

    saved = read_hub_api_url()
    table = Table(title="[brand]biolm hub status[/brand]", box=box.ROUNDED)
    table.add_column("Setting", style="brand")
    table.add_column("Value", style="text")
    table.add_row("Config path", str(config_path()))
    table.add_row("Saved hub API URL", saved or "[text.muted]not set[/text.muted]")
    table.add_row("Active model API URL", BIOLM_BASE_API_URL)
    table.add_row("Active source", get_model_api_source())
    console.print(table)

    if not saved and not is_hub_mode():
        console.print(
            "\n[text.muted]No hub configured. Run [brand]biolm hub set[/brand] "
            "after starting [brand]bh serve[/brand].[/text.muted]"
        )
        return

    probe_url = saved or BIOLM_BASE_API_URL
    with console.status("[brand]Probing hub...[/brand]"):
        status = fetch_hub_status(probe_url)

    console.print()
    if status.get("healthy"):
        origin = hub_origin(probe_url)
        console.print(
            f"[success]Hub is reachable[/success] — "
            f"{status.get('slug_count', 0)} variants, "
            f"{status.get('route_count', 0)} routes\n"
            f"Catalog: [brand]{origin}/catalog[/brand]"
        )
    else:
        console.print(
            f"[error]Hub not reachable[/error]: "
            f"{status.get('message') or 'unknown error'}"
        )


@hub.command("unset")
def hub_unset():
    """Remove saved hub settings and revert model inference to the hosted biolm.ai API.

    Has no effect on ``BIOLM_BASE_API_URL`` when that environment variable is set.
    """
    from biolm.hub.config import clear_hub_config

    if clear_hub_config():
        console.print(
            "[success]Removed hub configuration.[/success]\n"
            "[text.muted]Model API will use biolm.ai unless BIOLM_BASE_API_URL is set.[/text.muted]"
        )
    else:
        console.print("[text.muted]No hub configuration to remove.[/text.muted]")


@cli.group(cls=RichGroup)
def workspace():
    """List, inspect, create, and switch BioLM platform workspaces.

    A workspace is an account and environment pair, addressed as ``account/environment``.
    """
    pass


def _platform_request(callback):
    """Run one platform operation with deterministic client cleanup."""
    try:
        with PlatformClient() as client:
            return callback(client)
    except PlatformError as exc:
        raise click.ClickException(str(exc))


def _workspace_data(workspace_value: Workspace) -> Dict[str, Any]:
    """Return the stable public CLI representation of a workspace."""
    return {
        "path": workspace_value.path,
        "account_type": workspace_value.account_type,
        "account_id": workspace_value.account_id,
        "environment_id": workspace_value.environment_id,
    }


def _print_json(value: Any) -> None:
    """Write JSON without Rich markup or additional prose."""
    click.echo(json.dumps(value, indent=2, default=str))


def _identity_data(
    user: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, Any]:
    """Compose the stable public identity representation."""
    account_type = context.get("account_type")
    account_details = context.get("account_details") or {}
    is_personal = account_type == "user"
    return {
        "id": user.get("id"),
        "username": user.get("username"),
        "email": user.get("email"),
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "account_type": account_type,
        "account_id": context.get("account_id"),
        "account_name": None if is_personal else account_details.get("name"),
        "account_slug": (
            user.get("username") if is_personal else account_details.get("slug")
        ),
        "environment_id": context.get("environment_id"),
    }


def _identity_display_value(value: Any) -> str:
    return "—" if value is None or value == "" else str(value)


def _display_identity(data: Dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        _print_json(data)
        return

    display_name = " ".join(
        part
        for part in (data.get("first_name"), data.get("last_name"))
        if part
    )
    table = Table(
        title="[brand]Authenticated Identity[/brand]",
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bold",
    )
    table.add_column("Field", style="brand")
    table.add_column("Value")
    rows = (
        ("Username", data.get("username")),
        ("Email", data.get("email")),
        ("Display name", display_name),
        ("Account type", data.get("account_type")),
        ("Account ID", data.get("account_id")),
        ("Account name", data.get("account_name")),
        ("Account slug", data.get("account_slug")),
        ("Environment ID", data.get("environment_id")),
    )
    for label, value in rows:
        table.add_row(label, _identity_display_value(value))
    console.print(table)


@cli.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def whoami(output_format):
    """Show the authenticated principal and active account context."""
    data = _platform_request(
        lambda client: _identity_data(
            client.get_current_user(),
            client.get_context(),
        )
    )
    _display_identity(data, output_format)


def _display_workspace(workspace_value: Workspace, output_format: str) -> None:
    data = _workspace_data(workspace_value)
    if output_format == "json":
        _print_json(data)
        return

    table = Table(
        title="[brand]Workspace[/brand]",
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bold",
    )
    table.add_column("Path", style="brand")
    table.add_column("Account type")
    table.add_column("Account ID", justify="right")
    table.add_column("Environment ID", justify="right")
    table.add_row(
        str(data["path"]),
        str(data["account_type"]),
        str(data["account_id"]),
        str(data["environment_id"]),
    )
    console.print(table)


def _display_record(title: str, data: Dict[str, Any], output_format: str) -> None:
    """Display all fields returned by a platform endpoint."""
    if output_format == "json":
        _print_json(data)
        return

    table = Table(
        title="[brand]{}[/brand]".format(title),
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bold",
    )
    table.add_column("Field", style="brand")
    table.add_column("Value")
    for key, value in data.items():
        label = str(key).replace("_", " ").strip().title()
        if isinstance(value, (dict, builtins.list)):
            rendered = json.dumps(value, default=str)
        else:
            rendered = str(value)
        table.add_row(label, rendered)
    console.print(table)


@workspace.command("list")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def workspace_list(output_format):
    """List workspaces available to the authenticated user."""
    workspaces = _platform_request(lambda client: client.list_workspaces())
    data = [_workspace_data(item) for item in workspaces]
    if output_format == "json":
        _print_json(data)
        return

    table = Table(
        title="[brand]Workspaces[/brand]",
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bold",
    )
    table.add_column("Path", style="brand")
    table.add_column("Account type")
    table.add_column("Account ID", justify="right")
    table.add_column("Environment ID", justify="right")
    for item in data:
        table.add_row(
            str(item["path"]),
            str(item["account_type"]),
            str(item["account_id"]),
            str(item["environment_id"]),
        )
    console.print(table)


@workspace.command("show")
@click.argument("path", required=False)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def workspace_show(path, output_format):
    """Show the current workspace, or resolve an exact workspace PATH."""
    if path is None:
        workspace_value = _platform_request(lambda client: client.current_workspace())
    else:
        workspace_value = _platform_request(
            lambda client: client.get_workspace(path)
        )
    _display_workspace(workspace_value, output_format)


@workspace.command("switch")
@click.argument("path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def workspace_switch(path, output_format):
    """Switch the active account and environment to workspace PATH."""
    workspace_value = _platform_request(
        lambda client: client.switch_workspace(path)
    )
    _display_workspace(workspace_value, output_format)


@workspace.command("create")
@click.argument("name")
@click.option(
    "--account",
    "account_slug",
    help="Account slug in which to create the environment.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def workspace_create(name, account_slug, output_format):
    """Create a workspace environment named NAME."""
    workspace_value = _platform_request(
        lambda client: client.create_workspace(name, account=account_slug)
    )
    _display_workspace(workspace_value, output_format)


def _usage_display_value(value):
    """Render absent usage fields consistently."""
    return "—" if value is None or value == "" else str(value)


def _display_usage_summary(data: Dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        _print_json(data)
        return

    year = data.get("selected_year")
    month = data.get("selected_month")
    selected_month = (
        "{:04d}-{:02d}".format(int(year), int(month))
        if year is not None and month is not None
        else "—"
    )
    account = "{} {}".format(
        _usage_display_value(data.get("account_type")),
        _usage_display_value(data.get("account_id")),
    )
    filter_env_id = data.get("filter_env_id")
    environment = None
    if filter_env_id is not None:
        environment_label = data.get("environment_label")
        environment = "{} ({})".format(
            _usage_display_value(environment_label),
            filter_env_id,
        )

    summary = Table(
        title="[brand]Monthly usage[/brand]",
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bold",
    )
    summary.add_column("Field", style="brand")
    summary.add_column("Value")
    summary.add_row("Account", account)
    summary.add_row("Month", selected_month)
    summary.add_row("Environment filter", _usage_display_value(environment))
    summary.add_row(
        "Usage amount",
        _usage_display_value(data.get("current_usage_amount")),
    )
    summary.add_row(
        "Environment usage",
        _usage_display_value(data.get("environment_usage_amount")),
    )
    console.print(summary)

    model_charges = data.get("model_charges") or []
    if not model_charges:
        console.print("[text.muted]No model charges.[/text.muted]")
        return

    models = Table(
        title="[brand]Model charges[/brand]",
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bold",
    )
    models.add_column("Model", style="brand")
    models.add_column("Charge", justify="right")
    for item in model_charges:
        models.add_row(
            _usage_display_value(item.get("model_name")),
            _usage_display_value(item.get("total_biolm_charge")),
        )
    console.print(models)


@account.command("usage")
@click.option("--year", type=click.IntRange(min=1), help="Billing year.")
@click.option(
    "--month",
    type=click.IntRange(min=1, max=12),
    help="Billing month (1-12).",
)
@click.option(
    "--environment-id",
    type=click.IntRange(min=1),
    help="Filter to an environment ID.",
)
@click.option(
    "--account",
    help="Account slug (or personal label) whose usage to inspect.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def account_usage(year, month, environment_id, account, output_format):
    """Show monthly usage for the active or selected account."""
    data = _platform_request(
        lambda client: client.get_usage_summary(
            year=year,
            month=month,
            environment_id=environment_id,
            account=account,
        )
    )
    _display_usage_summary(data, output_format)


def _run_budget_show(output_format):
    """Show budget and usage fields for the active account."""
    data = _platform_request(lambda client: client.get_budget())
    _display_record("Account budget", data, output_format)


@account.group(cls=RichGroup, invoke_without_command=True)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.pass_context
def budget(ctx, output_format):
    """Inspect and set the active account budget.

    Invoked without a subcommand, shows the current budget.
    """
    if ctx.invoked_subcommand is None:
        _run_budget_show(output_format)


# Let negative numeric arguments reach FloatRange; extra unknown options still fail.
@budget.command("set", context_settings={"ignore_unknown_options": True})
@click.argument("amount", type=click.FloatRange(min=0.0))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def budget_set(amount, output_format):
    """Set the active account budget to nonnegative AMOUNT."""
    data = _platform_request(lambda client: client.set_budget(amount))
    _display_record("Account budget updated", data, output_format)


@click.command("show", cls=RichCommand)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def budget_show(output_format):
    """Show budget and usage fields for the active account."""
    _run_budget_show(output_format)


@account.group("api-key", cls=RichGroup)
def api_key():
    """Create and revoke BioLM platform API keys."""
    pass


@api_key.command("create")
@click.option(
    "--account",
    "account",
    help="Account slug (or personal label) that will own the key.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def api_key_create(account, output_format):
    """Create an API key for the active or selected account."""
    data = _platform_request(lambda client: client.create_api_key(account=account))
    if output_format == "json":
        _print_json(data)
        return

    _display_record("API key created", data, output_format)
    console.print(
        "[warning]Store this token now; it is shown only once.[/warning]"
    )


@api_key.command("delete")
@click.argument("token_or_prefix")
@click.option(
    "--yes",
    "assume_yes",
    is_flag=True,
    help="Skip the confirmation prompt.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def api_key_delete(token_or_prefix, assume_yes, output_format):
    """Revoke an API key by full token or eight-character prefix."""
    if not assume_yes:
        click.confirm("Revoke this API key?", abort=True)
    _platform_request(lambda client: client.delete_api_key(token_or_prefix))
    if output_format == "json":
        _print_json({"status": "deleted"})
        return
    console.print("[success]API key revoked.[/success]")


@account.group(cls=RichGroup)
def org():
    """List and manage BioLM organizations."""
    pass


@org.command("list")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def org_list(output_format):
    """List organizations available to the authenticated user."""
    organizations = _platform_request(lambda client: client.list_organizations())
    if output_format == "json":
        _print_json(organizations)
        return

    table = Table(
        title="[brand]Organizations[/brand]",
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bold",
    )
    table.add_column("ID", justify="right")
    table.add_column("Name")
    table.add_column("Slug", style="brand")
    for organization in organizations:
        table.add_row(
            str(organization.get("id", "")),
            str(organization.get("name", "")),
            str(organization.get("slug", "")),
        )
    console.print(table)


@org.command("show")
@click.argument("organization")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def org_show(organization, output_format):
    """Show organization by exact name or slug."""
    data = _platform_request(
        lambda client: client.get_organization(organization)
    )
    _display_record("Organization", data, output_format)


@org.command("invite")
@click.argument("organization")
@click.argument("email")
@click.option(
    "--role",
    type=click.Choice(["member", "admin", "billing_admin"]),
    default="member",
    show_default=True,
    help="Organization role.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def org_invite(organization, email, role, output_format):
    """Invite EMAIL to an organization by exact name or slug."""
    data = _platform_request(
        lambda client: client.invite_to_organization(
            organization, email, role=role
        )
    )
    _display_record("Organization invitation", data, output_format)


# Hidden compatibility aliases for pre-hierarchy command paths.
_usage_alias = RichGroup(
    "usage",
    hidden=True,
    help="Inspect monthly BioLM platform usage.",
)
cli.add_command(_usage_alias)
_hidden_leaf_alias(_usage_alias, "show", account_usage)

_budget_alias = RichGroup(
    "budget",
    hidden=True,
    help="Inspect and set the active account budget.",
)
cli.add_command(_budget_alias)
_hidden_leaf_alias(_budget_alias, "show", budget_show)
_hidden_leaf_alias(_budget_alias, "set", budget_set)

_apikey_alias = RichGroup(
    "apikey",
    hidden=True,
    help="Create and revoke BioLM platform API keys.",
)
cli.add_command(_apikey_alias)
_hidden_leaf_alias(_apikey_alias, "create", api_key_create)
_hidden_leaf_alias(_apikey_alias, "delete", api_key_delete)

_org_alias = RichGroup(
    "org",
    hidden=True,
    help="List and manage BioLM organizations.",
)
cli.add_command(_org_alias)
_hidden_leaf_alias(_org_alias, "list", org_list)
_hidden_leaf_alias(_org_alias, "show", org_show)
_hidden_leaf_alias(_org_alias, "invite", org_invite)


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
    """Browse the model catalog, inspect schemas and actions, and run inference from the terminal.

    Supports listing and filtering models, viewing metadata, and running encode, predict,
    fold, generate, and related actions against the hosted API or a connected biolm-hub.
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
    """List available models from the BioLM catalog with filtering, sorting, and export options.

    Output can be a Rich table in the terminal or saved as JSON, YAML, or CSV for scripting.
    
    Examples:

    .. code-block:: bash

        # List all models
        biolm model list

        # Filter for encoder models
        biolm model list --filter encoder=true

        # Sort by model name
        biolm model list --sort model_name

        # Output as JSON
        biolm model list --format json --output models.json

        # Compact view
        biolm model list --view compact
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
    """List the full open-source model catalog, including every deployable model on the platform.

    When connected to biolm-hub, lists gateway routes from OpenAPI instead of the hosted catalog.
    """
    if is_hub_mode():
        from biolm.hub.discovery import list_models_from_openapi
        from biolm.core.const import get_base_api_url

        models = list_models_from_openapi(get_base_api_url())
    else:
        from biolm.hub.catalog import list_catalog_models

        models = list_catalog_models()

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
    """Show metadata, actions, and optional JSON schemas for a specific model.

    Look up models by slug or display name; use ``--include-schemas`` for request/response shapes.
    
    Examples:

    .. code-block:: bash

        # Show model details
        biolm model show esm2-8m

        # Include schemas
        biolm model show esmfold --include-schemas

        # Output as JSON
        biolm model show esm2-8m --format json --output model.json
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
    """Run a model action against sequences or structures from files, stdin, or inline JSON.

    Supports encode, predict, generate, and lookup with batching, progress output, and
    common bioinformatics formats (FASTA, CSV, PDB, JSON).
    
    Examples:

    .. code-block:: bash

        # Run model with inline input
        echo '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}' | biolm model run esm2-8m encode -i - --format json

        # Run model with FASTA file
        biolm model run esmfold predict -i sequences.fasta -o results.json

        # Run with parameters
        biolm model run esm2-8m encode -i seq.fasta --params '{"normalize": true}'

        # Run with progress bar
        biolm model run esmfold predict -i large.fasta --progress
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
    """Generate copy-pasteable Python SDK examples for a model and action.

    Omit ``model_name`` to list available models; choose output as Python, Markdown, RST, or JSON.
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
    """Define, validate, run, and log multi-step BioLM protocol workflows.

    Protocols chain model calls and data transforms in YAML; use these commands to scaffold files,
    check schema compliance, inspect definitions, and send results to MLflow.
    """
    pass


def _protocol_request(callback):
    """Run one protocol API operation with consistent CLI errors."""
    try:
        return callback(ProtocolClient())
    except (ProtocolRunError, TimeoutError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc))


@protocol.command("list")
@click.option("--search", help="Filter protocols by name or slug.")
@click.option("--page", type=click.IntRange(min=1), default=1, show_default=True)
@click.option(
    "--page-size",
    type=click.IntRange(min=1, max=100),
    default=20,
    show_default=True,
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def protocol_list(search, page, page_size, output_format):
    """List protocols registered on the BioLM platform."""
    payload = _protocol_request(
        lambda client: client.list(search=search, page=page, page_size=page_size)
    )
    if output_format == "json":
        _print_json(payload)
        return

    protocols = payload.get("results") or []
    if not protocols:
        console.print("[text.muted]No protocols found.[/text.muted]")
        return

    table = Table(
        title="[brand]Protocols[/brand]",
        box=box.ROUNDED,
        show_header=True,
        header_style="brand.bold",
    )
    table.add_column("Slug", style="brand", no_wrap=True)
    table.add_column("Version", justify="right")
    table.add_column("Name")
    table.add_column("Details")
    for item in protocols:
        access = "public" if item.get("is_public") else "private"
        inputs = ", ".join(str(value) for value in item.get("input_fields") or [])
        details = "{} / {}".format(item.get("owner_type", ""), access)
        if inputs:
            details = "{}\ninputs: {}".format(details, inputs)
        table.add_row(
            str(item.get("slug", "")),
            str(item.get("version", "")),
            str(item.get("name", "")),
            details,
        )
    console.print(table)
    count = int(payload.get("count", len(protocols)))
    console.print(
        "[text.muted]Showing {} of {} protocol{}.[/text.muted]".format(
            len(protocols),
            count,
            "" if count == 1 else "s",
        )
    )


@protocol.command()
@click.argument('protocol_source', required=False)
def show(protocol_source):
    """Show a formatted report for a protocol from a local YAML file or platform ID.

    Displays tasks, dependencies, inputs/outputs, and configuration in a readable layout.
    
    Examples:

    .. code-block:: bash

        # Show protocol from YAML file
        biolm protocol show protocol.yaml

        # Show protocol from platform by ID
        biolm protocol show abc123
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


def _load_protocol_inputs(inputs_file) -> Dict[str, Any]:
    """Load a protocol input object from an optional JSON stream."""
    if inputs_file is None:
        return {}
    try:
        value = json.load(inputs_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException("Could not read protocol inputs as JSON: {}".format(exc))
    if not isinstance(value, dict):
        raise click.ClickException("Protocol inputs must be a JSON object.")
    return value


def _protocol_run_data(run) -> Dict[str, Any]:
    """Return the stable CLI summary of a submitted protocol run."""
    return {
        "run_id": run.run_id,
        "protocol_slug": run.protocol_slug,
        "protocol_version": run.protocol_version,
        "status": run.status,
    }


def _positive_float(ctx, param, value):
    """Validate positive protocol wait timing options."""
    if value is not None and value <= 0:
        raise click.BadParameter("must be greater than zero", ctx=ctx, param=param)
    return value


@protocol.command("run")
@click.argument("slug")
@click.option(
    "-i",
    "--inputs",
    "inputs_file",
    type=click.File("r"),
    help="JSON object containing protocol inputs. Use '-' for stdin.",
)
@click.option("--version", type=click.IntRange(min=1), help="Protocol version.")
@click.option("--name", "run_name", help="Human-readable run name.")
@click.option(
    "--environment-id",
    type=click.IntRange(min=1),
    help="Environment ID to attribute the run to.",
)
@click.option(
    "--wait",
    "wait_for_completion",
    is_flag=True,
    help="Wait for completion and print the full run result.",
)
@click.option(
    "--timeout",
    type=click.FLOAT,
    callback=_positive_float,
    default=3600.0,
    show_default=True,
    help="Total wait deadline in seconds.",
)
@click.option(
    "--poll-interval",
    type=click.FLOAT,
    callback=_positive_float,
    default=5.0,
    show_default=True,
    help="REST fallback polling interval in seconds.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def protocol_run(
    slug,
    inputs_file,
    version,
    run_name,
    environment_id,
    wait_for_completion,
    timeout,
    poll_interval,
    output_format,
):
    """Submit a run of registered protocol SLUG."""
    inputs = _load_protocol_inputs(inputs_file)
    run = _protocol_request(
        lambda client: client.submit(
            slug,
            inputs,
            version=version,
            run_name=run_name,
            environment_id=environment_id,
        )
    )

    if wait_for_completion:
        try:
            run.wait(
                timeout=timeout,
                show_progress=output_format != "json",
                poll_interval=poll_interval,
            )
            data = run.results()
        except (ProtocolRunError, TimeoutError, ValueError) as exc:
            raise click.ClickException(str(exc))
    else:
        data = _protocol_run_data(run)

    if output_format == "json":
        _print_json(data)
    else:
        _display_record("Protocol run", data, output_format)


@protocol.command("status")
@click.argument("run_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def protocol_status(run_id, output_format):
    """Show a current progress snapshot for protocol RUN_ID."""
    data = _protocol_request(lambda client: client.get_run(run_id).progress())
    _display_record("Protocol run status", data, output_format)


@protocol.command("wait")
@click.argument("run_id")
@click.option(
    "--timeout",
    type=click.FLOAT,
    callback=_positive_float,
    default=3600.0,
    show_default=True,
    help="Total wait deadline in seconds.",
)
@click.option(
    "--poll-interval",
    type=click.FLOAT,
    callback=_positive_float,
    default=5.0,
    show_default=True,
    help="REST fallback polling interval in seconds.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def protocol_wait(run_id, timeout, poll_interval, output_format):
    """Wait for protocol RUN_ID and print its final detail."""

    def wait_for_run(client):
        run = client.get_run(run_id)
        run.wait(
            timeout=timeout,
            show_progress=output_format != "json",
            poll_interval=poll_interval,
        )
        return run.results()

    data = _protocol_request(wait_for_run)
    _display_record("Protocol run", data, output_format)


@protocol.command("cancel")
@click.argument("run_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format.",
)
def protocol_cancel(run_id, output_format):
    """Request cancellation of protocol RUN_ID."""
    data = _protocol_request(lambda client: client.get_run(run_id).cancel())
    _display_record("Protocol cancellation", data, output_format)


@protocol.command("results")
@click.argument("run_id")
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    help="Write the full run detail as JSON instead of printing it.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="json",
    show_default=True,
    help="Terminal output format.",
)
def protocol_results(run_id, output, output_format):
    """Show or save results for protocol RUN_ID."""
    data = _protocol_request(lambda client: client.get_run(run_id).results())
    if output:
        try:
            Path(output).write_text(json.dumps(data, indent=2, default=str) + "\n")
        except OSError as exc:
            raise click.ClickException(str(exc))
        click.echo("Results written to {}".format(output))
        return
    _display_record("Protocol run results", data, output_format)


@protocol.command("download")
@click.argument("run_id")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default=".",
    show_default=True,
    help="Directory for the downloaded zip.",
)
@click.option(
    "--file-type",
    type=click.Choice(["csv", "jsonl"]),
    default="csv",
    show_default=True,
)
@click.option("--overwrite", is_flag=True, help="Replace an existing download.")
def protocol_download(run_id, output_dir, file_type, overwrite):
    """Download result artifacts for successful protocol RUN_ID."""
    path = _protocol_request(
        lambda client: client.get_run(run_id).download(
            output_dir=output_dir,
            file_type=file_type,
            overwrite=overwrite,
        )
    )
    click.echo("Downloaded results to {}".format(path))


@protocol.command()
@click.argument('protocol_file', type=click.Path(exists=True))
@click.option('--json', 'output_json', is_flag=True, help='Output results in JSON format')
def validate(protocol_file, output_json):
    """Validate a protocol YAML file against the JSON schema and internal rules.

    Checks YAML syntax, task references, circular dependencies, and template expressions.
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
    """Create a new protocol YAML file from a blank template or bundled example.

    Generated files can be validated immediately with ``biolm protocol validate``.
    
    Examples:

    .. code-block:: bash

        # Create a blank protocol
        biolm protocol init

        # Create with custom filename
        biolm protocol init my_protocol.yaml

        # Create from example
        biolm protocol init --example antibody_design

        # List available examples
        biolm protocol init --list-examples

        # Interactive mode
        biolm protocol init --interactive
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
    """Log protocol run results to MLflow using the protocol outputs configuration.

    Creates or updates an experiment at ``account/workspace/protocol`` and records metrics,
    parameters, and artifacts from a results file.
    
    Examples:

    .. code-block:: bash

        # Log results with outputs config from protocol file
        biolm protocol log results.jsonl --outputs protocol.yaml --account acme --workspace lab --protocol antifold-antibody

        # Dry run to see what would be logged
        biolm protocol log results.jsonl --outputs protocol.yaml --account acme --workspace lab --protocol antifold-antibody --dry-run

        # Use custom MLflow URI
        biolm protocol log results.jsonl --outputs protocol.yaml --account acme --workspace lab --protocol antifold-antibody --mlflow-uri http://localhost:5001
    """
    try:
        from biolm.plugins.mlflow.protocols import (
            MLflowNotAvailableError,
            log_protocol_results,
        )
    except ImportError:
        console.print(Panel(
            "[error]MLflow logging functionality is not available.[/error]\n\n"
            "Install MLflow support with: [brand]pip install biolm-sdk[mlflow][/brand]",
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
    """Create, inspect, and sync local datasets.

    Local datasets are directories with a ``dataset.yaml`` under configured roots
    (``~/.biolm/datasets``, ``./.biolm/datasets``). Use ``push`` / ``pull`` with
    ``--backend mlflow`` to sync via the optional MLflow plugin.
    """
    pass


def _dataset_client(root=None):
    from biolm.datasets import DatasetClient

    if root:
        return DatasetClient(roots=[root], primary_root=root)
    return DatasetClient()


@dataset.command("create")
@click.argument("dataset_id")
@click.option("--type", "dtype", default="files", help="Dataset type label (default: files)")
@click.option("--tag", multiple=True, help="Tag (repeatable)")
@click.option("--description", default=None, help="Human-readable description")
@click.option("--root", type=click.Path(), default=None, help="Root directory for the new dataset")
@click.option("--force", is_flag=True, help="Overwrite existing dataset.yaml")
def dataset_create(dataset_id, dtype, tag, description, root, force):
    """Create a new local dataset under the primary root.

    Examples:

    .. code-block:: bash

        biolm dataset create finetuning-v1 --tag finetune
        biolm dataset create my-set --root ./.biolm/datasets
    """
    from biolm.datasets import DatasetExistsError, DatasetError

    try:
        client = _dataset_client(root)
        ds = client.create(
            dataset_id,
            type=dtype,
            tags=builtins.list(tag) or None,
            description=description,
            root=root,
            force=force,
        )
        console.print(Panel(
            f"[success]Created dataset '{ds.id}'[/success]\n\n"
            f"Path: {ds.path}",
            title="[success]Dataset Created[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    except (DatasetExistsError, DatasetError) as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)


@dataset.command("init")
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--id", "dataset_id", required=True, help="Stable dataset id")
@click.option("--type", "dtype", default="files", help="Dataset type label (default: files)")
@click.option("--tag", multiple=True, help="Tag (repeatable)")
@click.option("--description", default=None, help="Human-readable description")
@click.option("--force", is_flag=True, help="Overwrite existing dataset.yaml")
def dataset_init(path, dataset_id, dtype, tag, description, force):
    """Adopt an existing directory as a dataset (writes dataset.yaml).

    Examples:

    .. code-block:: bash

        biolm dataset init ./training-data --id finetuning-v1 --tag finetune
    """
    from biolm.datasets import DatasetExistsError, DatasetError

    try:
        client = _dataset_client()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = client.init(
                path,
                id=dataset_id,
                type=dtype,
                tags=builtins.list(tag) or None,
                description=description,
                force=force,
            )
            for w in caught:
                console.print(f"[text.muted]{w.message}[/text.muted]")
        console.print(Panel(
            f"[success]Initialized dataset '{ds.id}'[/success]\n\n"
            f"Path: {ds.path}",
            title="[success]Dataset Initialized[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    except (DatasetExistsError, DatasetError) as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)


@dataset.command("list")
@click.option("--type", "dtype", default=None, help="Filter by dataset type")
@click.option("--tag", default=None, help="Filter by tag")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Save output to file")
def dataset_list(dtype, tag, fmt, output):
    """List local datasets under configured discovery roots.

    Examples:

    .. code-block:: bash

        biolm dataset list
        biolm dataset list --type files --tag finetune
        biolm dataset list --format json -o datasets.json
    """
    from biolm.datasets import DuplicateDatasetIdError

    try:
        client = _dataset_client()
        datasets = client.list(type=dtype, tag=tag)
    except DuplicateDatasetIdError as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Duplicate Dataset Id[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)

    rows = [ds.to_dict() for ds in datasets]

    if fmt == "json":
        output_data = json.dumps(rows, indent=2, default=str)
        if output:
            Path(output).write_text(output_data)
            console.print(f"[success]✓ Saved to {output}[/success]")
        else:
            click.echo(output_data)
        return

    if not datasets:
        console.print("[text.muted]No local datasets found.[/text.muted]\n")
        console.print(
            "Create one with: [brand]biolm dataset create my-dataset[/brand]"
        )
        return

    table = Table(
        title="[brand]Local Datasets[/brand]",
        show_header=True,
        header_style="brand.bold",
        box=box.ROUNDED,
    )
    table.add_column("ID", style="brand")
    table.add_column("Type", style="text")
    table.add_column("Tags", style="text.muted")
    table.add_column("Files", style="text.muted")
    table.add_column("Path", style="text")

    for ds in datasets:
        table.add_row(
            ds.id,
            ds.type,
            ", ".join(ds.tags) if ds.tags else "-",
            str(len(ds.files())),
            str(ds.path),
        )
    console.print(table)
    if output:
        Path(output).write_text(json.dumps(rows, indent=2, default=str))
        console.print(f"\n[success]✓ Saved to {output}[/success]")


@dataset.command("show")
@click.argument("id_or_path")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--output", "-o", type=click.Path(), help="Save output to file")
def dataset_show(id_or_path, fmt, output):
    """Show metadata and files for a local dataset by id or path.

    Examples:

    .. code-block:: bash

        biolm dataset show finetuning-v1
        biolm dataset show ./training-data
    """
    from biolm.datasets import DatasetNotFoundError, DuplicateDatasetIdError

    try:
        ds = _dataset_client().get(id_or_path)
    except (DatasetNotFoundError, DuplicateDatasetIdError) as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Dataset Not Found[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)

    data = ds.to_dict()
    if fmt == "json":
        output_data = json.dumps(data, indent=2, default=str)
        if output:
            Path(output).write_text(output_data)
            console.print(f"[success]✓ Saved to {output}[/success]")
        else:
            click.echo(output_data)
        return

    console.print(Panel(
        f"[brand]id[/brand]: {ds.id}\n"
        f"[brand]type[/brand]: {ds.type}\n"
        f"[brand]path[/brand]: {ds.path}\n"
        f"[brand]description[/brand]: {ds.description or '-'}\n"
        f"[brand]created_at[/brand]: {ds.created_at or '-'}\n"
        f"[brand]tags[/brand]: {', '.join(ds.tags) if ds.tags else '-'}",
        title=f"[brand]Dataset: {ds.id}[/brand]",
        border_style="brand",
        box=box.ROUNDED,
    ))

    files = ds.files()
    if files:
        table = Table(
            title="[brand]Files[/brand]",
            show_header=True,
            header_style="brand.bold",
            box=box.ROUNDED,
        )
        table.add_column("Path", style="brand")
        for rel in files:
            table.add_row(str(rel))
        console.print(table)
    else:
        console.print("[text.muted]No files yet. Add with: biolm dataset add "
                      f"{ds.id} FILE[/text.muted]")

    if output:
        Path(output).write_text(json.dumps(data, indent=2, default=str))
        console.print(f"\n[success]✓ Saved to {output}[/success]")


@dataset.command("add")
@click.argument("id_or_path")
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, help="Copy directories recursively")
def dataset_add(id_or_path, files, recursive):
    """Copy files into a local dataset.

    Examples:

    .. code-block:: bash

        biolm dataset add finetuning-v1 train.csv
        biolm dataset add ./my-set ./data --recursive
    """
    from biolm.datasets import DatasetError, DatasetNotFoundError, DuplicateDatasetIdError

    try:
        ds = _dataset_client().get(id_or_path)
        added = []
        for f in files:
            dest = ds.add(f, recursive=recursive)
            added.append(str(dest))
        console.print(Panel(
            f"[success]Added {len(added)} path(s) to '{ds.id}'[/success]\n\n"
            + "\n".join(added),
            title="[success]Files Added[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    except (DatasetNotFoundError, DatasetError, DuplicateDatasetIdError) as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)


@dataset.command("push")
@click.argument("id_or_path")
@click.option("--backend", required=True, help="Remote backend (e.g. mlflow)")
@click.option("--mlflow-uri", default="https://mlflow.biolm.ai/", help="MLflow tracking URI")
@click.option("--experiment", default=None, help="MLflow experiment name")
def dataset_push(id_or_path, backend, mlflow_uri, experiment):
    """Push a local dataset to a remote backend.

    Examples:

    .. code-block:: bash

        biolm dataset push finetuning-v1 --backend mlflow
    """
    from biolm.datasets import (
        BackendNotAvailableError,
        DatasetError,
        DatasetNotFoundError,
        DuplicateDatasetIdError,
    )

    try:
        if backend == "mlflow" and not are_credentials_valid():
            console.print(Panel(
                "[error]Authentication required.[/error]\n\n"
                "Please run [brand]biolm login[/brand] to authenticate.",
                title="[error]Not Authenticated[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)

        ds = _dataset_client().get(id_or_path)
        opts = {}
        if backend == "mlflow":
            opts["mlflow_uri"] = mlflow_uri
            opts["experiment_name"] = experiment
        with console.status(f"[brand]Pushing '{ds.id}' via {backend}...[/brand]"):
            result = ds.push(backend=backend, **opts)
        console.print(Panel(
            f"[success]✓ Pushed dataset '{ds.id}'[/success]\n\n"
            f"Backend: {backend}\n"
            f"Run ID: {result.get('run_id', 'N/A')}",
            title="[success]Push Complete[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    except (DatasetNotFoundError, BackendNotAvailableError, DatasetError, DuplicateDatasetIdError) as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            f"[error]Error pushing dataset: {e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)


@dataset.command("pull")
@click.argument("dataset_id")
@click.option("--backend", required=True, help="Remote backend (e.g. mlflow)")
@click.option("--path", "dest_path", type=click.Path(), default=None, help="Local destination directory")
@click.option("--force", is_flag=True, help="Overwrite conflicting local dataset.yaml")
@click.option("--mlflow-uri", default="https://mlflow.biolm.ai/", help="MLflow tracking URI")
@click.option("--experiment", default=None, help="MLflow experiment name")
def dataset_pull(dataset_id, backend, dest_path, force, mlflow_uri, experiment):
    """Pull a remote dataset into a local dataset directory.

    Defaults to ``~/.biolm/datasets/<id>/``.

    Examples:

    .. code-block:: bash

        biolm dataset pull finetuning-v1 --backend mlflow
        biolm dataset pull finetuning-v1 --backend mlflow --path ./my-copy
    """
    from biolm.datasets import (
        BackendNotAvailableError,
        DatasetError,
        DatasetExistsError,
    )

    try:
        if backend == "mlflow" and not are_credentials_valid():
            console.print(Panel(
                "[error]Authentication required.[/error]\n\n"
                "Please run [brand]biolm login[/brand] to authenticate.",
                title="[error]Not Authenticated[/error]",
                border_style="error",
                box=box.ROUNDED,
            ))
            sys.exit(1)

        opts = {"force": force}
        if backend == "mlflow":
            opts["mlflow_uri"] = mlflow_uri
            opts["experiment_name"] = experiment
        with console.status(f"[brand]Pulling '{dataset_id}' via {backend}...[/brand]"):
            ds = _dataset_client().pull(
                dataset_id,
                backend=backend,
                path=dest_path,
                **opts,
            )
        console.print(Panel(
            f"[success]✓ Pulled dataset '{ds.id}'[/success]\n\n"
            f"Path: {ds.path}\n"
            f"Backend: {backend}",
            title="[success]Pull Complete[/success]",
            border_style="success",
            box=box.ROUNDED,
        ))
    except (BackendNotAvailableError, DatasetExistsError, DatasetError) as e:
        console.print(Panel(
            f"[error]{e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)
    except Exception as e:
        console.print(Panel(
            f"[error]Error pulling dataset: {e}[/error]",
            title="[error]Error[/error]",
            border_style="error",
            box=box.ROUNDED,
        ))
        sys.exit(1)



if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
