"""The canonical shape of ingested race data.

Ingestion records what the timing feed says and nothing more: no laps are dropped here, no
values are imputed, and wet races are stored alongside dry ones with a flag. Filtering is a
modelling decision, so it lives in :mod:`apex.calibrate` where it can be justified, counted and
reported. A processed lap file is therefore a faithful, replayable copy of the source, and a
change to an exclusion rule never requires re-downloading anything.

Column names are snake_case and units are in the name (``lap_time_s``, ``track_temp_c``), so a
plot axis or a model coefficient can never silently disagree with the data about what it means.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Final

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "LAP_SCHEMA",
    "SCHEMA_VERSION",
    "EventMeta",
    "ScheduledRace",
    "SessionRecord",
    "conform_lap_frame",
    "event_slug",
]

SCHEMA_VERSION: Final = 1
"""Incremented whenever the meaning or membership of :data:`LAP_SCHEMA` changes.

Stored in every session sidecar. Ingestion re-runs a race whose stored version differs, which is
what stops a schema change from leaving stale files that look current.
"""

LAP_SCHEMA: Final[dict[str, pl.DataType]] = {
    # --- identity -------------------------------------------------------------------------
    "season": pl.Int16(),
    "round": pl.Int16(),
    "event": pl.String(),
    "circuit": pl.String(),
    "country": pl.String(),
    "driver": pl.String(),
    "driver_number": pl.String(),
    "team": pl.String(),
    # --- the lap --------------------------------------------------------------------------
    "lap": pl.Int16(),
    "stint": pl.Int16(),
    "compound": pl.String(),
    "tyre_life": pl.Int16(),
    "fresh_tyre": pl.Boolean(),
    "lap_time_s": pl.Float64(),
    "lap_start_s": pl.Float64(),
    "lap_end_s": pl.Float64(),
    "sector1_s": pl.Float64(),
    "sector2_s": pl.Float64(),
    "sector3_s": pl.Float64(),
    "speed_trap_kph": pl.Float64(),
    "position": pl.Int16(),
    "grid_position": pl.Int16(),
    # --- flags the exclusion rules read ---------------------------------------------------
    "track_status": pl.String(),
    "is_accurate": pl.Boolean(),
    "is_pit_in": pl.Boolean(),
    "is_pit_out": pl.Boolean(),
    "pit_in_s": pl.Float64(),
    "pit_out_s": pl.Float64(),
    "deleted": pl.Boolean(),
    "fastf1_generated": pl.Boolean(),
    # --- conditions -----------------------------------------------------------------------
    "air_temp_c": pl.Float64(),
    "track_temp_c": pl.Float64(),
    "rainfall": pl.Boolean(),
}
"""Canonical lap table: one row per driver per lap, in this order, with these types."""


def event_slug(event: str) -> str:
    """Filesystem-safe identifier for an event name.

    Example:
        >>> event_slug("Emilia Romagna Grand Prix")
        'emilia-romagna-grand-prix'
    """
    slug = re.sub(r"[^a-z0-9]+", "-", event.lower()).strip("-")
    if not slug:
        raise ValueError(f"event name {event!r} has no usable characters")
    return slug


class ScheduledRace(BaseModel):
    """Identity of one race weekend, as the published schedule reports it.

    Known before anything is downloaded, so it is what the CLI lists and iterates over.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    season: int
    round: int = Field(ge=1)
    event: str
    circuit: str
    country: str
    event_date: dt.date

    @property
    def slug(self) -> str:
        return event_slug(self.event)

    @property
    def key(self) -> str:
        """Stable file stem, sortable by season then round."""
        return f"{self.season}-{self.round:02d}-{self.slug}"


class EventMeta(ScheduledRace):
    """A scheduled race plus the facts only the loaded session knows."""

    total_laps: int = Field(ge=1)


class SessionRecord(BaseModel):
    """Sidecar written next to every ingested race.

    It answers, without opening the parquet: what this race is, how much of it survived the
    timing feed, whether it was dry, and exactly which code and configuration produced the file.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: EventMeta
    session: str = "R"
    n_laps: int = Field(ge=0)
    n_drivers: int = Field(ge=0)

    # --- dry-race classification (ADR-0002) ---------------------------------------------
    rainfall_fraction: float = Field(ge=0.0, le=1.0)
    """Fraction of session weather samples reporting rain."""

    wet_compound_laps: int = Field(ge=0)
    """Laps run on an intermediate or wet tyre by any driver."""

    is_dry: bool
    """True only if no wet tyre was fitted and rainfall stayed under the configured threshold."""

    # --- provenance ---------------------------------------------------------------------
    schema_version: int = SCHEMA_VERSION
    data_fingerprint: str
    """Hash of the ingestion-relevant configuration, from
    :meth:`~apex.config.DataSettings.ingestion_fingerprint`. A stored race whose fingerprint no
    longer matches is re-ingested, because its dry/wet classification was decided under rules
    that no longer apply."""

    apex_version: str
    fastf1_version: str
    config_hash: str
    """Full configuration hash, recorded for provenance only. It moves when a calibration knob
    moves, which must not invalidate a download."""

    ingested_at: dt.datetime

    @property
    def key(self) -> str:
        return self.event.key


def conform_lap_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Project a frame onto :data:`LAP_SCHEMA`, in order, with the declared types.

    Raises:
        ValueError: if a canonical column is missing. Ingestion fails loudly rather than
            emitting a file that is quietly missing a column a later phase depends on.
    """
    missing = [name for name in LAP_SCHEMA if name not in frame.columns]
    if missing:
        raise ValueError(f"lap frame is missing required columns: {missing}")
    return frame.select(
        pl.col(name).cast(dtype, strict=False) for name, dtype in LAP_SCHEMA.items()
    )
