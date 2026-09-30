"""Utilities for the transcription pipeline"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)


def _format_elapsed_time(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        A string like ``4.3s``, ``2m 07s``, or ``1h 02m 07s``, using the
        smallest unit combination that avoids leading zero components.
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes:
        return f"{minutes}m {seconds:02d}s"
    else:
        return f"{seconds:.1f}s"


@contextmanager
def _spinner(label: str) -> Generator[None, None, None]:
    """Context manager that shows an animated spinner while a block executes.

    Prints the label and elapsed time to stdout when the block exits.

    Args:
        label: Text displayed next to the spinner during execution.
    """
    start_time: float = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn(label),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task("", total=None)
        yield

    elapsed_time: float = time.monotonic() - start_time
    message = f"{label} ({_format_elapsed_time(elapsed_time)})"
    print(message)
    logger.info(message)
