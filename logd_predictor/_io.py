"""Shared Rich console and logging configuration for the package."""

import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console(highlight=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, show_path=False, show_time=True)],
)
