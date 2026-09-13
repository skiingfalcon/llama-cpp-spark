"""Two consoles, one rule: results go to stdout, everything else to stderr.

``out`` is for tables and answers a user might pipe somewhere; ``err`` is for progress,
warnings and errors. Modules import one (or both) instead of creating their own Console.
"""

from __future__ import annotations

from rich.console import Console

out = Console()
err = Console(stderr=True)
