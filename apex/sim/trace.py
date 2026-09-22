"""What a simulated race leaves behind.

One record per car per lap, and a classification at the end. The trace is the only output of the
simulator, and three very different consumers read it, which is why it is a flat table of facts
rather than a convenient object graph:

*   **Validation** compares it against the same quantities extracted from a real race, so every
    column it carries has to exist in the ingested lap schema under the same name and units.
*   **The front end** replays it, so it has to carry enough per-lap state -- position, gap, tyre,
    flags -- to animate a race without recomputing anything.
*   **Agent evaluation** reduces it to a scalar reward, so the finishing classification has to be
    unambiguous about who beat whom.

Columns follow the ingested lap schema's naming wherever the quantity is the same thing, so that a
simulated frame and a real frame can be compared column by column, and a plot of one can be a plot
of the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.sim.scenario import RaceScenario
from apex.sim.strategy import TrackPhase

__all__ = ["TRACE_SCHEMA", "LapRecord", "RaceResult", "RaceTrace", "StintRecord"]

TRACE_SCHEMA: dict[str, pl.DataType] = {
    "driver": pl.String(),
    "team": pl.String(),
    "lap": pl.Int16(),
    "position": pl.Int16(),
    "compound": pl.String(),
    "tyre_age": pl.Int16(),
    "stint": pl.Int16(),
    "lap_time_s": pl.Float64(),
    "elapsed_s": pl.Float64(),
    "gap_ahead_s": pl.Float64(),
    "gap_to_leader_s": pl.Float64(),
    "track_phase": pl.String(),
    "is_pit_lap": pl.Boolean(),
    "held_up": pl.Boolean(),
    "overtook": pl.Boolean(),
}
"""Canonical shape of a trace frame. Names match the ingested lap schema where the quantity is the
same, so simulated and real races are directly comparable."""


@dataclass(frozen=True, slots=True)
class LapRecord:
    """One car's one lap."""

    driver: str
    team: str
    lap: int
    position: int
    compound: str
    tyre_age: int
    stint: int
    lap_time_s: float
    elapsed_s: float
    gap_ahead_s: float
    gap_to_leader_s: float
    track_phase: TrackPhase
    is_pit_lap: bool
    held_up: bool
    """The car caught the one ahead and failed to pass, so it lost the time it would have gained.
    This is the cost of track position, and it is recorded because it is the quantity the whole
    strategic trade-off turns on."""

    overtook: bool


class StintRecord(BaseModel):
    """One car's run on one set of tyres."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver: str
    stint: int
    compound: str
    first_lap: int
    last_lap: int

    @property
    def laps(self) -> int:
        return self.last_lap - self.first_lap + 1


class RaceResult(BaseModel):
    """One car's finishing classification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    driver: str
    team: str
    grid_position: int
    finish_position: int = Field(ge=1)
    total_time_s: float
    gap_to_winner_s: float = Field(ge=0.0)
    stops: int = Field(ge=0)
    laps_held_up: int = Field(ge=0)
    overtakes: int = Field(ge=0)

    @property
    def positions_gained(self) -> int:
        """Places made up from the grid. Negative means places lost."""
        return self.grid_position - self.finish_position


class RaceTrace(BaseModel):
    """A complete simulated race."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    scenario: RaceScenario
    results: tuple[RaceResult, ...]
    stints: tuple[StintRecord, ...]
    laps: tuple[LapRecord, ...] = Field(repr=False)
    safety_car_laps: tuple[int, ...]
    virtual_safety_car_laps: tuple[int, ...]
    seed_run_id: str
    """Identifier of the seed bundle this race was drawn from, so it can be reproduced exactly."""

    @property
    def winner(self) -> RaceResult:
        return self.results[0]

    def result(self, driver: str) -> RaceResult:
        for entry in self.results:
            if entry.driver == driver:
                return entry
        raise KeyError(f"{driver!r} did not race here")

    def finishing_order(self) -> tuple[str, ...]:
        return tuple(entry.driver for entry in self.results)

    def frame(self) -> pl.DataFrame:
        """The per-lap records as a typed table, in the canonical trace schema."""
        if not self.laps:
            return pl.DataFrame(schema=TRACE_SCHEMA)
        rows = [
            {
                "driver": record.driver,
                "team": record.team,
                "lap": record.lap,
                "position": record.position,
                "compound": record.compound,
                "tyre_age": record.tyre_age,
                "stint": record.stint,
                "lap_time_s": record.lap_time_s,
                "elapsed_s": record.elapsed_s,
                "gap_ahead_s": record.gap_ahead_s,
                "gap_to_leader_s": record.gap_to_leader_s,
                "track_phase": str(record.track_phase),
                "is_pit_lap": record.is_pit_lap,
                "held_up": record.held_up,
                "overtook": record.overtook,
            }
            for record in self.laps
        ]
        return pl.DataFrame(rows, schema=TRACE_SCHEMA).sort("lap", "position")

    def results_frame(self) -> pl.DataFrame:
        return pl.DataFrame([entry.model_dump() for entry in self.results]).sort("finish_position")


@dataclass(slots=True)
class _TraceBuilder:
    """Accumulates records during a race. Not part of the public surface."""

    records: list[LapRecord] = field(default_factory=list)
    safety_car_laps: list[int] = field(default_factory=list)
    virtual_safety_car_laps: list[int] = field(default_factory=list)

    def add(self, record: LapRecord) -> None:
        self.records.append(record)

    def note_phase(self, lap: int, phase: TrackPhase) -> None:
        if phase is TrackPhase.SAFETY_CAR:
            self.safety_car_laps.append(lap)
        elif phase is TrackPhase.VIRTUAL_SAFETY_CAR:
            self.virtual_safety_car_laps.append(lap)
