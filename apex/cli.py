"""Command-line entry point.

Kept deliberately thin: commands resolve configuration and delegate. Each phase adds its own
sub-command alongside a Makefile target.
"""

from __future__ import annotations

import json

import polars as pl
import typer

from apex import __version__
from apex.calibrate.dataset import training_laps
from apex.calibrate.exclusions import clean_laps
from apex.calibrate.params import ParameterSet, calibrate, params_path, schema_path, write_schema
from apex.calibrate.report import render_report
from apex.config import AppSettings, get_settings
from apex.data import SessionRecord, ingest_seasons, load_laps, manifest_frame
from apex.data.sources import RateLimitError
from apex.eval import validate_holdout
from apex.sim import RaceSimulator, ReactiveRules, from_grid
from apex.sim.scenario import AnchorMode

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
    try:
        records = ingest_seasons(seasons, settings, events=events, force=force, on_progress=report)
    except RateLimitError as limit:
        stored = manifest_frame(settings)
        typer.echo(f"\nTiming API rate limit reached ({limit}).")
        typer.echo(
            f"{stored.height} races are on disk and nothing was lost. "
            "Re-run the same command once the quota resets to continue where this stopped."
        )
        raise typer.Exit(code=75) from limit  # EX_TEMPFAIL: retry later, nothing is wrong

    dry = sum(1 for record in records if record.is_dry)
    typer.echo(f"{len(records)} races ingested this run, {dry} dry.")


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


@app.command(name="calibrate")
def calibrate_command(
    report: bool = typer.Option(True, help="Also write the human-readable calibration report."),
) -> None:
    """Fit every simulator parameter from the training seasons and write `params/params.json`.

    The training seasons are the only source. A number in that file that came from a holdout race
    would invalidate every metric this project publishes, so the split is enforced here, by the
    loader, rather than trusted downstream.
    """
    settings = get_settings()
    data = training_laps(settings)
    typer.echo(
        f"Fitting from {data.n_races} dry races at {data.n_circuits} circuits, "
        f"seasons {list(settings.data.train_seasons)}."
    )

    params = calibrate(settings, data)
    destination = params_path(settings)
    params.write(destination)
    write_schema(settings)

    typer.echo(_table(params.diagnostics_frame()))
    typer.echo(f"\nWrote {destination} and {schema_path(settings)}")

    if report:
        report_path = settings.paths.figures.parent / "calibration_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_report(params), encoding="utf-8")
        typer.echo(f"Wrote {report_path}")


@app.command()
def params() -> None:
    """Summarise the fitted parameters currently on disk."""
    settings = get_settings()
    path = params_path(settings)
    if not path.is_file():
        typer.echo("No parameters fitted yet. Run `apex calibrate`.")
        raise typer.Exit(code=1)
    typer.echo(render_report(ParameterSet.read(path)))


@app.command()
def race(
    *,
    circuit: str = typer.Option(..., help="Circuit to race at, as the parameter file names it."),
    laps: int = typer.Option(..., help="Race distance."),
    season: int = typer.Option(
        2025, help="Season, used to place the pace level on the fitted trend."
    ),
    drivers: str = typer.Option(
        "VER,NOR,LEC,PIA,SAI,RUS,HAM,PER,ALO,STR", help="Starting grid, front to back."
    ),
    stops: int = typer.Option(1, help="Planned stops for the rule-based pit wall."),
    episode: int = typer.Option(
        0, help="Episode index; changes the race without changing the seed."
    ),
) -> None:
    """Simulate one race from a grid and print the result.

    Everything comes from `params/params.json`: the circuit's pace level, each car's pace from its
    grid slot, tyre degradation, pit loss, the safety-car hazard and the passing model.
    """
    settings = get_settings()
    path = params_path(settings)
    if not path.is_file():
        typer.echo("No parameters fitted yet. Run `apex calibrate`.")
        raise typer.Exit(code=1)

    parameters = ParameterSet.read(path)
    grid = [name.strip().upper() for name in drivers.split(",") if name.strip()]
    try:
        scenario = from_grid(
            parameters,
            circuit=circuit,
            season=season,
            total_laps=laps,
            grid=grid,
            settings=settings,
        )
    except KeyError:
        known = ", ".join(sorted(entry.circuit for entry in parameters.baseline.circuits))
        typer.echo(f"No baseline fitted for circuit {circuit!r}.\nFitted circuits: {known}")
        raise typer.Exit(code=1) from None

    simulator = RaceSimulator(parameters, settings)
    bundle = settings.seeds().derive("race").derive(circuit).episode(episode)
    trace = simulator.run(scenario, ReactiveRules(target_stops=stops), bundle)

    typer.echo(
        f"{circuit} {season}, {laps} laps -- reference lap "
        f"{scenario.anchor.reference_lap_s:.3f}s ({scenario.anchor.mode})"
    )
    if trace.safety_car_laps:
        typer.echo(f"Safety car: laps {list(trace.safety_car_laps)}")
    if trace.virtual_safety_car_laps:
        typer.echo(f"Virtual safety car: laps {list(trace.virtual_safety_car_laps)}")
    typer.echo(_table(trace.results_frame().drop("team")))
    typer.echo(f"run_id={trace.seed_run_id}")


@app.command()
def validate(
    free_runs: int = typer.Option(
        0, help="Seeds per race in free mode. Zero runs replay mode only, which is much faster."
    ),
    limit: int | None = typer.Option(None, help="Validate only the first N holdout races."),
    per_race: bool = typer.Option(False, help="Also print a row per race."),
) -> None:
    """Measure the simulator against the held-out races, and print the error.

    This is the command that decides whether the project may call itself a digital twin. It refuses
    to run if the parameter file was fitted on a season being validated, because then every number
    it printed would be meaningless.
    """
    settings = get_settings()
    typer.echo(
        f"Validating on holdout seasons {list(settings.data.holdout_seasons)}; "
        f"parameters fitted on {list(settings.data.train_seasons)}."
    )
    try:
        report = validate_holdout(settings, free_runs=free_runs, limit=limit)
    except ValueError as error:
        typer.echo(f"\n{error}")
        raise typer.Exit(code=1) from error

    typer.echo(f"{report.n_races} races.\n")
    header = (
        f"{'mode':<22}{'posMAE':>8}{'grid':>8}"
        f"{'spearman':>10}{'winner':>9}{'lapMAE':>9}{'gapMAE':>9}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for anchor in (AnchorMode.ANCHORED, AnchorMode.FITTED):
        for source in ("grid", "oracle"):
            summary = report.summary(anchor, pace_source=source)
            if not summary:
                continue
            typer.echo(
                f"{f'{anchor}/{source}':<22}"
                f"{summary['position_mae']:>8.2f}"
                f"{summary['grid_position_mae']:>8.2f}"
                f"{summary['order_spearman']:>10.3f}"
                f"{summary['winner_accuracy']:>8.0%}"
                f"{summary['lap_time_mae_s']:>9.2f}"
                f"{summary['finish_gap_mae_s']:>9.1f}"
            )
    typer.echo(
        "\nposMAE is the twin's mean finishing-position error. `grid` is the same error for a model"
        "\nthat predicts everybody finishes where they qualified -- the twin has to beat it to mean"
        "\nanything. A row marked `oracle` was handed each car's true race pace: it is a diagnostic"
        "\nand never a prediction, and its distance from `grid` is the cost of forecasting pace."
    )

    if per_race:
        frame = report.replay_frame().filter(
            (pl.col("anchor_mode") == str(AnchorMode.ANCHORED)) & (pl.col("pace_source") == "grid")
        )
        typer.echo(
            "\n"
            + _table(
                frame.select(
                    "event",
                    "n_compared",
                    "n_retired_excluded",
                    "position_mae",
                    "grid_position_mae",
                    "order_spearman",
                    "lap_time_mae_s",
                    "finish_gap_mae_s",
                ).sort("position_mae")
            )
        )

    if report.free:
        typer.echo("\n" + _table(report.free_frame()))


if __name__ == "__main__":  # pragma: no cover
    app()
