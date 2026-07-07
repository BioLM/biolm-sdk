"""Shared Rich progress UI for batch operations. Used by CLI and Model (progress=True)."""
from contextlib import contextmanager
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
    _RICH_AVAILABLE = True
except ImportError:
    Console = None  # type: ignore[misc, assignment]
    _RICH_AVAILABLE = False


@contextmanager
def rich_progress(
    total_items: int,
    description: str = "Processing...",
    console: Optional[Any] = None,
):
    """
    Context manager that yields a progress callback for batch operations.

    The callback has signature (completed: int, total: int) -> None.
    Call it after each batch with the number of items completed so far and total items.
    Uses Rich Progress (spinner, bar, task progress) when Rich is available;
    otherwise yields a no-op callback.

    Args:
        total_items: Total number of items to process.
        description: Description shown in the progress bar.
        console: Rich Console instance (default: new Console()).

    Yields:
        A callable (completed: int, total: int) -> None to update progress.
    """
    if not _RICH_AVAILABLE or total_items <= 0:
        yield lambda completed, total: None
        return

    if console is None:
        console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(description, total=total_items)

        def callback(completed: int, total: int) -> None:
            progress.update(task_id, completed=completed)

        yield callback
