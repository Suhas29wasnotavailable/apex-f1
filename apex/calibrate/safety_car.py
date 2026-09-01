"""How often a race is neutralised, for how long, and where.

This is the parameter the whole project turns on. Under deterministic dynamics a strategy can be
solved exactly in advance; it is the arrival of a safety car -- unpredictable, and worth twenty
seconds of pit loss when it comes -- that makes the problem stochastic, and stochasticity is the
only condition under which reinforcement learning can earn its place here. A hazard rate that is
wrong makes the entire comparison meaningless, in either direction.

Full and virtual safety cars are estimated separately, because strategically they are opposites.
A full safety car bunches the field and cuts the cost of a stop to almost nothing. A virtual one
slows everybody in place and does neither. Averaging them into one "neutralisation" rate would
produce a number that describes no real event.

Deployments are counted, not laps. A safety car that runs for four laps is one event with a
duration, not four independent chances of an event, and counting laps would inflate the hazard at
exactly the circuits where safety cars last longest.

Rates are estimated, pooled and shrunk entirely on the log-odds scale, and transformed back to
probabilities only at the end. Doing the pooling on the probability scale would be simpler and
would produce intervals reaching below zero, which is not a statement about safety cars.

Safety cars are rare enough that a circuit can easily have seen none at all, and the raw
proportion then says a safety car at that circuit is impossible -- a point estimate of zero, an
interval of zero width, and a log-odds of negative infinity. The Jeffreys prior supplies half a
notional deployment and half a notional clear lap, which keeps every rate strictly inside
``(0, 1)``. Half rather than the more common two: with roughly sixty laps at risk per race and a
true rate near one in eighty, two notional deployments would nearly double the estimate at a
circuit seen twice. The raw counts are reported alongside every rate so the unadjusted proportion
can always be recovered.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.calibrate.estimates import Estimate
from apex.calibrate.pooling import partial_pool
from apex.config import AppSettings, get_settings

__all__ = [
    "Neutralisation",
    "NeutralisationModel",
    "SafetyCarModel",
    "fit_safety_car",
    "neutralisation_laps",
]

FULL = "safety_car"
VIRTUAL = "virtual_safety_car"

_PSEUDO_COUNT: Final = 0.5
"""The Jeffreys prior's half a notional deployment and half a notional clear lap."""


class Neutralisation(BaseModel):
    """The chance a neutralisation begins on any given green lap, at one circuit or pooled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    circuit: str | None = None
    per_lap_probability: Estimate
    n_deployments: int = Field(ge=0)
    laps_at_risk: int = Field(ge=0)
    """Green laps that could have started one: every lap whose predecessor was not neutralised."""

    n_races: int = Field(ge=0)

    @property
    def expected_per_race(self) -> float:
        """Deployments expected in a race of average length at this circuit."""
        if self.n_races == 0:
            return 0.0
        return self.per_lap_probability.value * (self.laps_at_risk / self.n_races)


class NeutralisationModel(BaseModel):
    """One kind of neutralisation, everywhere it could be measured."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    pooled: Neutralisation
    circuits: tuple[Neutralisation, ...]
    duration_laps: Estimate | None
    """Mean laps a deployment lasts, pooled across circuits. The simulator needs this to know how
    long the pit-loss discount stays open. ``None`` when this kind of neutralisation never
    happened in the training seasons, which is a fact about the data rather than a duration."""

    races_with_any: int = Field(ge=0)
    n_races: int = Field(ge=0)

    @property
    def share_of_races(self) -> float:
        return self.races_with_any / self.n_races if self.n_races else 0.0

    def circuit(self, name: str) -> Neutralisation:
        for entry in self.circuits:
            if entry.circuit == name:
                return entry
        raise KeyError(f"no hazard fitted for circuit {name!r}")


class SafetyCarModel(BaseModel):
    """Both kinds of neutralisation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    safety_car: NeutralisationModel
    virtual_safety_car: NeutralisationModel


def neutralisation_laps(all_laps: pl.DataFrame, codes: tuple[str, ...]) -> pl.DataFrame:
    """Reduce the field's per-driver track status to one row per race lap.

    Args:
        all_laps: Featured, unfiltered laps -- ``CalibrationData.all_laps``.
        codes: Track status codes that count as this kind of neutralisation.

    Returns:
        ``race_id``, ``circuit``, ``lap``, ``active``, ``starts`` and ``at_risk``, one row per lap
        of each race. A lap is active if any car on track saw the status during it, since the
        status is a property of the track rather than of a car.
    """
    if not codes:
        raise ValueError("at least one track status code is required")
    matches = pl.any_horizontal(
        [pl.col("track_status").fill_null("").str.contains(code, literal=True) for code in codes]
    )
    per_lap = (
        all_laps.group_by(["race_id", "circuit", "lap"])
        .agg(matches.any().alias("active"))
        .sort("race_id", "lap")
    )
    # The lap before the first is green by definition: a race starts under green flags, so a
    # first-lap safety car is a deployment and lap one is a lap at risk like any other.
    previous = pl.col("active").shift(1).over("race_id").fill_null(value=False)
    return per_lap.with_columns(
        (pl.col("active") & ~previous).alias("starts"),
        (~previous).alias("at_risk"),
    ).with_columns((pl.col("active") & ~previous).cum_sum().over("race_id").alias("deployment"))


def _log_odds_hazard(deployments: int, laps_at_risk: int, n_races: int, level: float) -> Estimate:
    """Per-lap hazard on the log-odds scale, where it is pooled and shrunk.

    Degrees of freedom come from the number of races rather than the number of laps: laps within a
    race are anything but independent draws -- one incident neutralises several of them -- and
    using the lap count would produce an interval several times too narrow.
    """
    adjusted_trials = laps_at_risk + 2.0 * _PSEUDO_COUNT
    probability = (deployments + _PSEUDO_COUNT) / adjusted_trials
    log_odds = float(np.log(probability / (1.0 - probability)))
    std_error = float(np.sqrt(1.0 / (adjusted_trials * probability * (1.0 - probability))))
    return Estimate.from_standard_error(
        log_odds,
        std_error,
        n_obs=laps_at_risk,
        level=level,
        degrees_of_freedom=float(max(n_races - 1, 1)),
    )


def _from_log_odds(estimate: Estimate) -> Estimate:
    """Map an estimate on the log-odds scale back to a probability.

    The interval transforms endpoint by endpoint, which is what keeps it inside ``(0, 1)``; the
    standard error transforms by the delta method, for readers who want one.
    """

    def expit(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-value)))

    probability = expit(estimate.value)
    scale = probability * (1.0 - probability)
    return estimate.model_copy(
        update={
            "value": probability,
            "std_error": None if estimate.std_error is None else estimate.std_error * scale,
            "interval": (
                None
                if estimate.interval is None
                else (expit(estimate.interval[0]), expit(estimate.interval[1]))
            ),
        }
    )


def _unmeasured_duration(durations: np.ndarray, level: float) -> Estimate | None:
    """One deployment gives a length but no spread; none gives nothing at all."""
    if durations.size == 0:
        return None
    return Estimate(value=float(durations[0]), level=level, n_obs=1)


def _durations(laps: pl.DataFrame) -> np.ndarray:
    """Length in laps of every deployment."""
    active = laps.filter(pl.col("active"))
    if active.height == 0:
        return np.zeros(0)
    counts = active.group_by(["race_id", "deployment"]).len()
    return counts["len"].to_numpy().astype(np.float64)


def _fit_kind(
    all_laps: pl.DataFrame, kind: str, codes: tuple[str, ...], settings: AppSettings
) -> NeutralisationModel:
    level = settings.calibration.confidence_level
    laps = neutralisation_laps(all_laps, codes)
    n_races = laps.select("race_id").n_unique()

    per_race = laps.group_by("race_id").agg(pl.col("starts").sum().alias("starts"))
    total_deployments = int(laps["starts"].sum())
    total_at_risk = int(laps["at_risk"].sum())
    pooled_log_odds = _log_odds_hazard(total_deployments, total_at_risk, n_races, level)
    pooled = Neutralisation(
        kind=kind,
        circuit=None,
        per_lap_probability=_from_log_odds(pooled_log_odds),
        n_deployments=total_deployments,
        laps_at_risk=total_at_risk,
        n_races=n_races,
    )

    minimum = settings.calibration.min_races_per_circuit
    by_circuit = laps.group_by("circuit").agg(
        pl.col("starts").sum().alias("deployments"),
        pl.col("at_risk").sum().alias("laps_at_risk"),
        pl.n_unique("race_id").alias("races"),
    )
    observed: dict[str, Estimate] = {}
    facts: dict[str, tuple[int, int, int]] = {}
    for circuit, deployments, laps_at_risk, races in by_circuit.iter_rows():
        facts[str(circuit)] = (int(deployments), int(laps_at_risk), int(races))
        if int(races) >= minimum:
            observed[str(circuit)] = _log_odds_hazard(
                int(deployments), int(laps_at_risk), int(races), level
            )

    # Shrink toward the rate the whole calendar shows, not toward the average of twenty sparse
    # circuits: each circuit's Jeffreys half-deployment would otherwise be counted twenty times.
    shrunk = (
        partial_pool(observed, level=level, toward=pooled_log_odds).estimates if observed else {}
    )
    circuits = tuple(
        Neutralisation(
            kind=kind,
            circuit=name,
            per_lap_probability=_from_log_odds(
                shrunk.get(
                    name,
                    pooled_log_odds.model_copy(update={"shrinkage": 0.0, "n_obs": facts[name][1]}),
                )
            ),
            n_deployments=facts[name][0],
            laps_at_risk=facts[name][1],
            n_races=facts[name][2],
        )
        for name in sorted(facts)
    )

    durations = _durations(laps)
    duration = (
        Estimate.from_standard_error(
            float(durations.mean()),
            float(durations.std(ddof=1) / np.sqrt(durations.size)),
            n_obs=int(durations.size),
            level=level,
        )
        if durations.size > 1
        else _unmeasured_duration(durations, level)
    )
    return NeutralisationModel(
        kind=kind,
        pooled=pooled,
        circuits=circuits,
        duration_laps=duration,
        races_with_any=int((per_race["starts"] > 0).sum()),
        n_races=n_races,
    )


def fit_safety_car(all_laps: pl.DataFrame, settings: AppSettings | None = None) -> SafetyCarModel:
    """Estimate both neutralisation hazards, their durations, and their circuit variation.

    Args:
        all_laps: Featured, unfiltered laps -- ``CalibrationData.all_laps``. The pace filter
            removes exactly the laps this measurement is made from.
        settings: Configuration; the process-wide settings by default.
    """
    settings = settings or get_settings()
    calibration = settings.calibration
    return SafetyCarModel(
        safety_car=_fit_kind(all_laps, FULL, calibration.full_safety_car_status, settings),
        virtual_safety_car=_fit_kind(
            all_laps, VIRTUAL, calibration.virtual_safety_car_status, settings
        ),
    )
