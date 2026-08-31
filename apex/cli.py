"""Command-line entry point.

Kept deliberately thin: commands resolve configuration and delegate. Each phase adds its own
sub-command alongside a Makefile target.
"""

from __future__ import annotations

import json

import typer

from apex import __version__
from apex.config import AppSettings, get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="APEX -- F1 race digital twin.")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def config(paths: bool = typer.Option(False, help="Include machine-specific paths.")) -> None:
    """Print the resolved configuration and its provenance hash."""
    settings: AppSettings = get_settings()
    payload = (
        settings.model_dump(mode="json")
        if paths
        else settings.model_dump(mode="json", exclude={"paths"})
    )
    typer.echo(json.dumps({"config_hash": settings.config_hash(), **payload}, indent=2))


@app.command()
def seed() -> None:
    """Print the master seed and the run identifier it implies."""
    settings = get_settings()
    bundle = settings.seeds()
    typer.echo(f"master_seed={bundle.master_seed} run_id={bundle.run_id}")


if __name__ == "__main__":  # pragma: no cover
    app()
