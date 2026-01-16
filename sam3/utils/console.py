# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Styled console output for SAM3 debugging and logging."""

from rich.console import Console as RichConsole


class StyledConsole:
    """Console wrapper with styled output methods for debugging.

    Usage:
        from sam3.utils import console

        console.info("tag", "Processing started")
        console.success("tag", "Feature enabled")
        console.skip("tag", "Feature disabled")
        console.warning("tag", "Something unexpected")
        console.error("tag", "Something failed")
    """

    def __init__(self) -> None:
        self._console = RichConsole()

    def print(self, *args, **kwargs) -> None:
        """Pass-through to rich console print."""
        self._console.print(*args, **kwargs)

    def info(self, tag: str, message: str) -> None:
        """Print info message with cyan tag."""
        self._console.print(f"[cyan][{tag}][/cyan] {message}")

    def success(self, tag: str, message: str) -> None:
        """Print success message with green checkmark."""
        self._console.print(f"[cyan][{tag}][/cyan] [green]✓[/green] {message}")

    def skip(self, tag: str, message: str) -> None:
        """Print skipped/inactive message in dim."""
        self._console.print(f"[cyan][{tag}][/cyan] [dim]○ {message}[/dim]")

    def warning(self, tag: str, message: str) -> None:
        """Print warning message in yellow."""
        self._console.print(f"[cyan][{tag}][/cyan] [yellow]⚠ {message}[/yellow]")

    def error(self, tag: str, message: str) -> None:
        """Print error message in red."""
        self._console.print(f"[cyan][{tag}][/cyan] [red]✗ {message}[/red]")


# Global instance
console = StyledConsole()
