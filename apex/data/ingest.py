"""Turning race sessions into files on disk, once.

The processed store is content-addressed by race, not by run: ``laps/2023-01-bahrain-grand-prix
.parquet`` beside ``sessions/2023-01-bahrain-grand-prix.json``. One file per race rather than one
appended table means an interrupted season pull resumes exactly where it stopped, a single
corrupt session can be re-fetched on its own, and the manifest can be rebuilt by listing a
directory instead of being maintained as mutable state.

Re-ingestion is skipped when the stored :data:`~apex.data.schema.SCHEMA_VERSION` still matches,
so the expensive part -- the download -- happens once per race for as long as the schema holds.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from apex import __version__
from apex.config import AppSettings, get_settings
from apex.data.schema import (
    LAP_SCHEMA,
    SCHEMA_VERSION,
    ScheduledRace,
    SessionRecord,
    conform_lap_frame,
)
from apex.data.sources import (
    RawRace,
    fastf1_version,
    fetch_race,
    race_calendar,
    scheduled_race,
    to_lap_frame,
)

__all__ = [
    "classify_dryness",
    "ingest_race",
    "ingest_seasons",
    "laps_path",
    "load_laps",
    "load_manifest",
    "manifest_frame",
    "record_path",
    "summarise",
]


def laps_path(settings: AppSettings, key: str) -> Path:
    return settings.paths.processed / "laps" / f"{key}.parquet"


def record_path(settings: AppSettings, key: str) -> Path:
    return settings.paths.processed / "sessions" / f"{key}.json"


def classify_dryness(
    settings: AppSettings, laps: pl.DataFrame, rainfall_fraction: float
) -> tuple[int, bool]:
    """Apply ADR-0002's dry-race test.

    A race is dry only if no driver fitted an intermediate or wet tyre *and* rain was reported in
    no more than ``max_rainfall_fraction`` of weather samples. The tyre test is the stricter of
    the two in practice: teams fit wets for conditions the weather station under-reports.

    Returns:
        The number of laps run on a wet compound, and whether the race counts as dry.
    """
    wet = settings.data.wet_compounds
    wet_laps = int(laps.filter(pl.col("compound").is_in(wet)).height)
    is_dry = wet_laps == 0 and rainfall_fraction <= settings.data.max_rainfall_fraction
    return wet_laps, is_dry


def summarise(settings: AppSettings, raw: RawRace, laps: pl.DataFrame) -> SessionRecord:
    """Build the sidecar describing one ingested race."""
    wet_laps, is_dry = classify_dryness(settings, laps, raw.rainfall_fraction)
    return SessionRecord(
        event=raw.event,
        session=settings.data.session,
        n_laps=laps.height,
        n_drivers=int(laps["driver"].n_unique()),
        rainfall_fraction=raw.rainfall_fraction,
        wet_compound_laps=wet_laps,
        is_dry=is_dry,
        schema_version=SCHEMA_VERSION,
        data_fingerprint=settings.data.ingestion_fingerprint(),
        apex_version=__version__,
        fastf1_version=fastf1_version(),
        config_hash=settings.config_hash(),
        ingested_at=dt.datetime.now(dt.UTC),
    )


def _stored_record(path: Path, fingerprint: str) -> SessionRecord | None:
    """Return a usable sidecar, or None if it is absent, unreadable, or out of date.

    Out of date means one of two things: the lap schema has changed since the file was written,
    or the ingestion-relevant configuration has. Either way the stored race is stale, and the
    dangerous outcome -- a stale file that still looks current -- is what this prevents. A
    sidecar this version of the code cannot parse is treated the same way, as absent.
    """
    if not path.is_file():
        return None
    try:
        record = SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError:
        return None
    if record.schema_version != SCHEMA_VERSION or record.data_fingerprint != fingerprint:
        return None
    return record


def ingest_race(
    season: int,
    event: str | int,
    settings: AppSettings | None = None,
    *,
    force: bool = False,
) -> SessionRecord:
    """Ingest one race, or return the existing record if it is already current.

    Args:
        season: Championship year.
        event: Event name or round number.
        settings: Configuration; the process-wide settings by default.
        force: Re-fetch and rewrite even when a current file exists.
    """
    settings = settings or get_settings()

    # Resolve identity from the schedule, which is one cheap cached request, so that a race
    # already on disk costs nothing further. Without this, resuming an interrupted season pull
    # would re-read every session it had already stored.
    key = scheduled_race(season, event, settings.paths.fastf1_cache).key
    if not force:
        existing = _stored_record(record_path(settings, key), settings.data.ingestion_fingerprint())
        if existing is not None and laps_path(settings, key).is_file():
            return existing

    raw = fetch_race(season, event, settings.paths.fastf1_cache)
    if raw.event.key != key:
        raise ValueError(f"schedule resolved {key} but the session loaded {raw.event.key}")
    laps = to_lap_frame(raw)
    record = summarise(settings, raw, laps)
    write_race(settings, record, laps)
    return record


def write_race(settings: AppSettings, record: SessionRecord, laps: pl.DataFrame) -> None:
    """Write a race's lap table and sidecar, creating directories as needed."""
    parquet = laps_path(settings, record.key)
    sidecar = record_path(settings, record.key)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    conform_lap_frame(laps).write_parquet(parquet, compression="zstd")
    sidecar.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def planned_races(
    seasons: Sequence[int], settings: AppSettings, events: Sequence[str] | None = None
) -> tuple[ScheduledRace, ...]:
    """Every race to be ingested for the given seasons, optionally narrowed to named events."""
    wanted = {name.casefold() for name in events} if events else None
    planned: list[ScheduledRace] = []
    for season in seasons:
        for race in race_calendar(season, settings.paths.fastf1_cache):
            if wanted is None or race.event.casefold() in wanted:
                planned.append(race)
    return tuple(planned)


def ingest_seasons(
    seasons: Sequence[int],
    settings: AppSettings | None = None,
    *,
    events: Sequence[str] | None = None,
    force: bool = False,
    on_progress: Callable[[SessionRecord], None] | None = None,
) -> tuple[SessionRecord, ...]:
    """Ingest whole seasons, skipping races that are already current.

    Args:
        seasons: Championship years.
        settings: Configuration; the process-wide settings by default.
        events: Restrict to these event names (case-insensitive), e.g. the pilot races.
        force: Re-fetch every race even if a current file exists.
        on_progress: Optional callable invoked with each :class:`SessionRecord` as it lands, so
            a long pull can report progress without this module knowing about the console.
    """
    settings = settings or get_settings()
    records: list[SessionRecord] = []
    for race in planned_races(seasons, settings, events):
        record = ingest_race(race.season, race.event, settings, force=force)
        records.append(record)
        if on_progress is not None:
            on_progress(record)
    return tuple(records)


def load_manifest(settings: AppSettings | None = None) -> tuple[SessionRecord, ...]:
    """Every ingested race on disk, in season and round order."""
    settings = settings or get_settings()
    directory = settings.paths.processed / "sessions"
    if not directory.is_dir():
        return ()
    records = [
        SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]
    return tuple(sorted(records, key=lambda record: (record.event.season, record.event.round)))


def manifest_frame(settings: AppSettings | None = None) -> pl.DataFrame:
    """The manifest as a flat table, for reporting and joins."""
    records = load_manifest(settings)
    if not records:
        return pl.DataFrame(schema={"key": pl.String()})
    return pl.DataFrame(
        [
            {
                "key": record.key,
                "season": record.event.season,
                "round": record.event.round,
                "event": record.event.event,
                "circuit": record.event.circuit,
                "n_laps": record.n_laps,
                "n_drivers": record.n_drivers,
                "total_laps": record.event.total_laps,
                "rainfall_fraction": record.rainfall_fraction,
                "wet_compound_laps": record.wet_compound_laps,
                "is_dry": record.is_dry,
            }
            for record in records
        ]
    )


def load_laps(
    settings: AppSettings | None = None,
    *,
    seasons: Iterable[int] | None = None,
    dry_only: bool | None = None,
) -> pl.DataFrame:
    """Read ingested laps back as one table.

    Args:
        settings: Configuration; the process-wide settings by default.
        seasons: Restrict to these seasons. Calibration passes the training seasons here, which
            is the mechanism that keeps the holdout season out of every fitted parameter.
        dry_only: Keep only races classified dry. Defaults to the configured ``data.dry_only``.

    Returns:
        The concatenated lap table, conforming to :data:`~apex.data.schema.LAP_SCHEMA`. Empty
        with the correct schema if nothing matches, so downstream code needs no special case.
    """
    settings = settings or get_settings()
    keep_dry = settings.data.dry_only if dry_only is None else dry_only
    wanted_seasons = set(seasons) if seasons is not None else None

    paths = [
        laps_path(settings, record.key)
        for record in load_manifest(settings)
        if (not keep_dry or record.is_dry)
        and (wanted_seasons is None or record.event.season in wanted_seasons)
    ]
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return pl.DataFrame(schema=LAP_SCHEMA)
    frames = [pl.read_parquet(path) for path in existing]
    return conform_lap_frame(pl.concat(frames, how="vertical")).sort(
        "season", "round", "driver", "lap"
    )
