"""Command-line entry point.

Kept deliberately thin: commands resolve configuration and delegate. Each phase adds its own
sub-command alongside a Makefile target.
"""

from __future__ import annotations

import json

import polars as pl
import typer

from apex import __version__
from apex.calibrate.exclusions import clean_laps
from apex.config import AppSettings, get_settings
from apex.data import SessionRecord, ingest_seasons, load_laps, manifest_frame

app = typer.Typer(add_completion=False, no_args_is_help=True, help="APEX -- F1 race digital twin.")


def _table(frame: pl.DataFrame) -> str:
    """Render a frame in full: a truncated diagnostic is a misleading diagnostic."""
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, tbl_hide_dataframe_shape=True):
        return str(frame)


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


@app.command()
def ingest(
    season: list[int] = typer.Option(  # noqa: B008 - typer reads defaults at decoration time
        None, "--season", help="Season to ingest. Repeatable. Defaults to every configured season."
    ),
    event: list[str] = typer.Option(  # noqa: B008
        None, "--event", help="Event name to ingest. Repeatable. Defaults to the whole calendar."
    ),
    pilot: bool = typer.Option(False, help="Restrict to the configured pilot events."),
    force: bool = typer.Option(False, help="Re-fetch and rewrite races already on disk."),
) -> None:
    """Download race sessions and write canonical lap tables.

    Safe to interrupt and re-run: races already stored under the current schema are skipped.
    """
    settings = get_settings()
    seasons = tuple(season) if season else settings.data.all_seasons
    events = tuple(event) if event else (settings.data.pilot_events if pilot else None)

    def report(record: SessionRecord) -> None:
        state = "dry" if record.is_dry else "WET (excluded from calibration)"
        typer.echo(f"  {record.key}  {record.n_laps:>5} laps  {record.n_drivers:>2} cars  {state}")

    typer.echo(
        f"Ingesting seasons {list(seasons)}" + (f", events {list(events)}" if events else "")
    )
    records = ingest_seasons(seasons, settings, events=events, force=force, on_progress=report)
    dry = sum(1 for record in records if record.is_dry)
    typer.echo(f"{len(records)} races on disk, {dry} dry.")


@app.command()
def races() -> None:
    """List the ingested races and their dry/wet classification.

    Wet races are stored and listed, not hidden: which races the calibration cannot use is part
    of the result, and a reader should be able to count them.
    """
    frame = manifest_frame(get_settings())
    if frame.height == 0:
        typer.echo("No races ingested yet. Run `apex ingest --pilot` to start.")
        raise typer.Exit(code=1)
    typer.echo(
        _table(frame.select("season", "round", "event", "circuit", "n_laps", "n_drivers", "is_dry"))
    )
    dry = int(frame["is_dry"].sum())
    typer.echo(f"{frame.height} races ingested, {dry} dry, {frame.height - dry} wet.")


@app.command()
def laps(
    season: list[int] = typer.Option(  # noqa: B008
        None, "--season", help="Restrict to these seasons. Defaults to the training seasons."
    ),
    wet: bool = typer.Option(False, help="Include races classified wet."),
) -> None:
    """Report what the exclusion rules keep, and what they cost.

    This is the diagnostic that has to be read before any fitted number is believed: it shows how
    many real laps each rule removed and how many survive per compound.
    """
    settings = get_settings()
    seasons = tuple(season) if season else settings.data.train_seasons
    frame = load_laps(settings, seasons=seasons, dry_only=not wet)
    if frame.height == 0:
        typer.echo(f"No ingested laps for seasons {list(seasons)}. Run `apex ingest` first.")
        raise typer.Exit(code=1)

    report = clean_laps(frame, settings)
    races_seen = frame.select("season", "round").unique().height
    typer.echo(f"{races_seen} races, seasons {list(seasons)}")
    typer.echo(_table(report.diagnostics))
    typer.echo(f"retained {report.retained_fraction:.1%} of laps")
    typer.echo(
        _table(
            report.laps.group_by("compound")
            .agg(
                pl.len().alias("laps"),
                pl.col("lap_time_s").median().round(3).alias("median_s"),
                pl.n_unique("circuit").alias("circuits"),
            )
            .sort("compound")
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
