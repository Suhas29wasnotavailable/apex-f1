"""Ingestion: real race sessions in, canonical lap tables on disk out.

This layer is deliberately free of modelling judgement. It records what the timing feed reported,
flags each race dry or wet, and stamps every file with the code and configuration that produced
it. Which laps deserve to inform a parameter is decided in :mod:`apex.calibrate`.
"""

from __future__ import annotations

from apex.data.ingest import (
    ingest_race,
    ingest_seasons,
    laps_path,
    load_laps,
    load_manifest,
    manifest_frame,
    record_path,
)
from apex.data.schema import (
    LAP_SCHEMA,
    SCHEMA_VERSION,
    EventMeta,
    ScheduledRace,
    SessionRecord,
    conform_lap_frame,
    event_slug,
)

__all__ = [
    "LAP_SCHEMA",
    "SCHEMA_VERSION",
    "EventMeta",
    "ScheduledRace",
    "SessionRecord",
    "conform_lap_frame",
    "event_slug",
    "ingest_race",
    "ingest_seasons",
    "laps_path",
    "load_laps",
    "load_manifest",
    "manifest_frame",
    "record_path",
]
