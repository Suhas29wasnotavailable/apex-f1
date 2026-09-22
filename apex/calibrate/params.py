"""The parameter file: everything the simulator is allowed to know, and where it came from.

``params/params.json`` is the only channel between calibration and the rest of the project. Nothing
downstream reads a lap; the simulator reads this file, and every number in it arrived by the same
route -- fitted from training-season laps, carrying an interval, stamped with the configuration and
code that produced it.

That constraint is what makes the honesty claims checkable rather than rhetorical. Because the file
records its seasons, a reader can confirm no holdout race touched it. Because it records the config
hash and the seed, the run can be repeated. Because every value is an
:class:`~apex.calibrate.estimates.Estimate`, no parameter can be published without its uncertainty.
And because the exclusion diagnostics travel in the same file, the laps that were thrown away are
as visible as the ones that were kept.

The JSON schema is written alongside and committed, so a consumer -- the simulator, the API, the
frontend -- can validate the file it is handed instead of trusting its shape.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Final

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex import __version__
from apex.calibrate.baseline import BaselineModel, car_pace_frame, fit_baseline
from apex.calibrate.dataset import CalibrationData, training_laps
from apex.calibrate.grid import GridPaceModel, fit_grid_pace
from apex.calibrate.overtaking import OvertakingModel, fit_overtaking
from apex.calibrate.pace import CircuitPace, PaceModel, fit_circuit_pace
from apex.calibrate.pit import PitLossModel, fit_pit_loss
from apex.calibrate.safety_car import SafetyCarModel, fit_safety_car
from apex.config import AppSettings, get_settings

__all__ = [
    "PARAMS_SCHEMA_VERSION",
    "ExclusionCount",
    "ParameterSet",
    "Provenance",
    "calibrate",
    "params_path",
    "schema_path",
    "write_schema",
]

PARAMS_SCHEMA_VERSION: Final = 2
"""Incremented when the shape of the parameter file changes.

Version 2 adds everything P2's simulator needs and P1's estimators had no reason to publish: the
absolute pace level the car-race fixed effects absorbed (:mod:`~apex.calibrate.baseline`), the
mapping from a starting grid to race pace that lets a field be built from pre-race information
alone (:mod:`~apex.calibrate.grid`), and the passing model without which cars can only change
position in the pit lane (:mod:`~apex.calibrate.overtaking`).
"""


class Provenance(BaseModel):
    """Where a parameter file came from, in enough detail to reproduce or refute it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: dt.datetime
    apex_version: str
    config_hash: str
    seed: int
    schema_version: int = PARAMS_SCHEMA_VERSION

    seasons: tuple[int, ...]
    """Seasons the fit was allowed to see. ADR-0002's temporal split is auditable from here: if a
    holdout season appears in this list, every number in the file is contaminated."""

    holdout_seasons: tuple[int, ...]
    """Seasons deliberately withheld, recorded so the split is visible without the config."""

    n_races: int
    n_circuits: int
    n_laps_available: int
    n_laps_used: int

    @property
    def retained_fraction(self) -> float:
        return self.n_laps_used / self.n_laps_available if self.n_laps_available else 0.0


class ExclusionCount(BaseModel):
    """One row of the exclusion audit: what a rule removed and what survived it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    removed: int = Field(ge=0)
    remaining: int = Field(ge=0)


class ParameterSet(BaseModel):
    """Every fitted parameter, with its provenance and the filtering that produced it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provenance: Provenance
    exclusions: tuple[ExclusionCount, ...]

    pace: PaceModel
    """The pace model fitted to every training lap at once."""

    circuit_pace: tuple[PaceModel, ...]
    """One partially-pooled pace model per circuit."""

    between_circuit_std: dict[str, float]
    """Spread of each pace coefficient across circuits. A zero says the circuits disagreed no
    more than their own noise explains -- a finding, not a missing value."""

    pit_loss: PitLossModel
    safety_car: SafetyCarModel

    baseline: BaselineModel
    """Absolute pace: the circuit lap-time level, the season trend, and the field's spread. The
    pace models above are all differences; this is what they are differences from."""

    grid_pace: GridPaceModel
    """Race pace as a function of starting position, so a field can be constructed from pre-race
    information alone. Validation depends on this: a field built from the held-out race's own lap
    times would be a restatement of the answer rather than a prediction of it."""

    overtaking: OvertakingModel
    """Pass probability given gap and pace advantage, with per-circuit difficulty."""

    def diagnostics_frame(self) -> pl.DataFrame:
        """The exclusion audit as a table, for display."""
        return pl.DataFrame([entry.model_dump() for entry in self.exclusions])

    def circuit_pace_at(self, circuit: str) -> PaceModel:
        for model in self.circuit_pace:
            if model.circuit == circuit:
                return model
        raise KeyError(f"no pace model for circuit {circuit!r}")

    def write(self, path: Path) -> None:
        """Write the parameter file, creating its directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> ParameterSet:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()


def params_path(settings: AppSettings) -> Path:
    return settings.paths.params / "params.json"


def schema_path(settings: AppSettings) -> Path:
    return settings.paths.params / "params.schema.json"


def _exclusions(diagnostics: pl.DataFrame) -> tuple[ExclusionCount, ...]:
    return tuple(
        ExclusionCount(rule=str(rule), removed=int(removed), remaining=int(remaining))
        for rule, removed, remaining in diagnostics.iter_rows()
    )


def calibrate(
    settings: AppSettings | None = None, data: CalibrationData | None = None
) -> ParameterSet:
    """Fit every parameter from the training seasons.

    Args:
        settings: Configuration; the process-wide settings by default.
        data: Prepared laps. Loaded from the training seasons by default, which is the only
            source a published parameter file may have.

    Raises:
        ValueError: if there are no laps to fit from. An empty parameter file would be worse
            than none, because it would look like a result.
    """
    settings = settings or get_settings()
    data = data if data is not None else training_laps(settings)
    if data.laps.height == 0:
        raise ValueError(
            f"no laps ingested for seasons {list(settings.data.train_seasons)}; "
            f"run `apex ingest` first"
        )

    pace: CircuitPace = fit_circuit_pace(data.laps, settings)

    # Everything below the pace model is fitted on the corrected laps the pace model implies, so
    # the per-circuit models are passed through rather than the pooled one: a car's residual pace
    # should not carry the pooled degradation curve's error at a circuit where a better curve was
    # fitted. The order matters -- baseline defines the pace scale, and the grid and overtaking
    # models are both expressed on it.
    cars = car_pace_frame(data.laps, pace.circuits, settings)

    return ParameterSet(
        provenance=Provenance(
            generated_at=dt.datetime.now(dt.UTC),
            apex_version=__version__,
            config_hash=settings.config_hash(),
            seed=settings.seed,
            seasons=data.seasons,
            holdout_seasons=settings.data.holdout_seasons,
            n_races=data.n_races,
            n_circuits=data.n_circuits,
            n_laps_available=data.all_laps.height,
            n_laps_used=data.laps.height,
        ),
        exclusions=_exclusions(data.diagnostics),
        pace=pace.pooled,
        circuit_pace=tuple(pace.circuits[name] for name in sorted(pace.circuits)),
        between_circuit_std=pace.between_std,
        pit_loss=fit_pit_loss(data.all_laps, settings),
        safety_car=fit_safety_car(data.all_laps, settings),
        baseline=fit_baseline(data.laps, pace.circuits, settings, all_laps=data.all_laps),
        grid_pace=fit_grid_pace(cars, settings),
        overtaking=fit_overtaking(data.all_laps, cars, pace.circuits, settings),
    )


def write_schema(settings: AppSettings) -> Path:
    """Write the committed JSON schema for the parameter file."""
    path = schema_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ParameterSet.json_schema(), indent=2) + "\n", encoding="utf-8")
    return path
