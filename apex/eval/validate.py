"""Validating the twin against races it has never seen.

This is the module the word "twin" has to earn. ADR-0002 withheld the whole of 2025 from
calibration; this runs the simulator on those races and publishes how far off it was, in modes
chosen so that a good number cannot come from having been told the answer.

**Replay.** The simulator is given the real grid, the real pit stops and the real safety cars, and
produces the racing. This is the sharpest test of the pace, degradation and traffic models, because
strategy and luck are held fixed: if the finishing order still comes out wrong, the fault is in how
the cars were modelled and nowhere else.

**Free.** The simulator is given the grid and nothing else. Safety cars are drawn, a rule-based pit
wall decides the stops, and the race runs many times from different seeds. This tests the whole twin
at once, and it is the mode the front end runs in. Reported as a distribution, because a single free
race is one sample from a stochastic process and quoting one would be meaningless.

Both modes are run under both anchors -- pace level taken from the race's own median lap time, and
pace level predicted from the fitted circuit baseline and season trend -- because the difference
between those two columns is exactly the cost of not knowing how fast the cars would be in a season
the calibration never saw. Reporting only the anchored column would overstate the twin; reporting
only the fitted one would blame the race model for a pace forecast.

**The baseline every number is read against is the starting grid.** A model that predicts everybody
finishes where they qualified is surprisingly good, and any twin that cannot beat it is not adding
anything. It is computed over the same drivers, in every mode, and printed next to the twin's own
error.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.calibrate.dataset import prepare_laps
from apex.calibrate.params import ParameterSet, params_path
from apex.config import AppSettings, get_settings
from apex.eval.metrics import RaceComparison, compare
from apex.eval.observed import ObservedRace, ReplayStops, observed_pace, observed_race
from apex.seeding import SeedBundle
from apex.sim.race import RaceSimulator
from apex.sim.scenario import AnchorMode, from_race
from apex.sim.strategies import ReactiveRules
from apex.sim.trace import RaceTrace

__all__ = [
    "FreeSummary",
    "ValidationReport",
    "validate_holdout",
    "validate_race",
]

DEFAULT_FREE_RUNS = 30
"""Seeds per race in free mode. Enough to see the spread of outcomes without turning validation into
a training run."""


class FreeSummary(BaseModel):
    """Free-mode results for one race, across seeds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: str
    anchor_mode: str
    n_runs: int = Field(ge=1)
    position_mae_mean: float
    position_mae_std: float
    grid_position_mae: float
    winner_hit_rate: float
    """Share of seeds in which the simulated winner was the real winner."""

    podium_overlap_mean: float
    safety_car_rate: float
    """Share of seeds that produced at least one full safety car. Comparable against whether the
    real
    race had one, which is the crudest check that the hazard is not wildly wrong."""

    real_had_safety_car: bool


class ValidationReport(BaseModel):
    """Everything one validation run measured."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seasons: tuple[int, ...]
    n_races: int
    replay: tuple[RaceComparison, ...]
    free: tuple[FreeSummary, ...]
    params_config_hash: str
    params_seasons: tuple[int, ...]
    """Seasons the parameters were fitted on. If a validated season appears here the result is void,
    so it travels with the report rather than being asserted in prose."""

    def replay_frame(self) -> pl.DataFrame:
        if not self.replay:
            return pl.DataFrame()
        return pl.DataFrame([entry.model_dump() for entry in self.replay])

    def free_frame(self) -> pl.DataFrame:
        if not self.free:
            return pl.DataFrame()
        return pl.DataFrame([entry.model_dump() for entry in self.free])

    def summary(
        self, anchor_mode: AnchorMode = AnchorMode.ANCHORED, *, pace_source: str = "grid"
    ) -> dict[str, float]:
        """Headline replay numbers for one anchor and pace source, averaged over races."""
        rows = [
            entry
            for entry in self.replay
            if entry.anchor_mode == str(anchor_mode) and entry.pace_source == pace_source
        ]
        if not rows:
            return {}
        return {
            "races": float(len(rows)),
            "position_mae": float(np.mean([row.position_mae for row in rows])),
            "grid_position_mae": float(np.mean([row.grid_position_mae for row in rows])),
            "order_spearman": float(np.mean([row.order_spearman for row in rows])),
            "winner_accuracy": float(np.mean([row.winner_correct for row in rows])),
            "podium_overlap": float(np.mean([row.podium_overlap for row in rows])),
            "lap_time_mae_s": float(
                np.mean([row.lap_time_mae_s for row in rows if row.lap_time_mae_s is not None])
            ),
            "finish_gap_mae_s": float(
                np.mean([row.finish_gap_mae_s for row in rows if row.finish_gap_mae_s is not None])
            ),
        }


def _replay_once(
    simulator: RaceSimulator,
    laps: pl.DataFrame,
    observed: ObservedRace,
    *,
    anchor_mode: AnchorMode,
    bundle: SeedBundle,
    pace_overrides: dict[str, float] | None = None,
) -> RaceTrace:
    """One replay: real grid, real stops, real flags, simulated racing.

    ``pace_overrides`` replaces the grid model's pace forecast, and is only ever supplied by the
    oracle diagnostic.
    """
    scenario = from_race(
        simulator.params,
        laps,
        anchor_mode=anchor_mode,
        settings=simulator.settings,
        pace_overrides=pace_overrides,
    )
    return simulator.run(
        scenario,
        ReplayStops(stops=observed.stops),
        bundle,
        schedule=observed.schedule,
    )


def _free_summary(
    simulator: RaceSimulator,
    laps: pl.DataFrame,
    observed: ObservedRace,
    *,
    anchor_mode: AnchorMode,
    bundle: SeedBundle,
    runs: int,
) -> FreeSummary:
    """Many free races, summarised."""
    scenario = from_race(
        simulator.params, laps, anchor_mode=anchor_mode, settings=simulator.settings
    )
    strategy = ReactiveRules(target_stops=1, compounds=("HARD",))

    comparisons: list[RaceComparison] = []
    with_safety_car = 0
    for index in range(runs):
        trace = simulator.run(scenario, strategy, bundle.episode(index))
        comparisons.append(compare(trace, observed))
        with_safety_car += int(bool(trace.safety_car_laps))

    errors = np.array([entry.position_mae for entry in comparisons], dtype=np.float64)
    return FreeSummary(
        event=observed.event,
        anchor_mode=str(anchor_mode),
        n_runs=runs,
        position_mae_mean=float(errors.mean()),
        position_mae_std=float(errors.std(ddof=1)) if errors.size > 1 else 0.0,
        grid_position_mae=comparisons[0].grid_position_mae,
        winner_hit_rate=float(np.mean([entry.winner_correct for entry in comparisons])),
        podium_overlap_mean=float(np.mean([entry.podium_overlap for entry in comparisons])),
        safety_car_rate=with_safety_car / runs,
        real_had_safety_car=bool(observed.safety_car_laps),
    )


def validate_race(
    simulator: RaceSimulator,
    laps: pl.DataFrame,
    bundle: SeedBundle,
    *,
    free_runs: int = DEFAULT_FREE_RUNS,
    anchors: Sequence[AnchorMode] = (AnchorMode.ANCHORED, AnchorMode.FITTED),
    oracle: bool = True,
) -> tuple[list[RaceComparison], list[FreeSummary]]:
    """Validate one race in every mode and under every anchor.

    Args:
        simulator: Simulator built on the fitted parameters.
        laps: Featured, unfiltered laps of exactly one race.
        bundle: Seed bundle for this race.
        free_runs: Seeds per race in free mode. Zero skips free mode.
        anchors: Anchors to run under.
        oracle: Also run an anchored replay with each car's true race pace substituted in. Labelled
            ``pace_source="oracle"``, never a prediction, and the reason it is on by default is that
            without it the honest numbers cannot be interpreted -- it separates the error the race
            model is responsible for from the error the pre-race pace forecast is responsible for.

    Returns:
        The replay comparisons and the free-mode summaries for this race.
    """
    observed = observed_race(laps, simulator.settings)
    replays: list[RaceComparison] = []
    frees: list[FreeSummary] = []

    for anchor_mode in anchors:
        replay = _replay_once(
            simulator,
            laps,
            observed,
            anchor_mode=anchor_mode,
            bundle=bundle.derive(f"replay/{anchor_mode}"),
        )
        replays.append(compare(replay, observed))
        if free_runs > 0:
            frees.append(
                _free_summary(
                    simulator,
                    laps,
                    observed,
                    anchor_mode=anchor_mode,
                    bundle=bundle.derive(f"free/{anchor_mode}"),
                    runs=free_runs,
                )
            )

    if oracle:
        overrides = observed_pace(laps, simulator.params, simulator.settings)
        if overrides:
            oracle_trace = _replay_once(
                simulator,
                laps,
                observed,
                anchor_mode=AnchorMode.ANCHORED,
                bundle=bundle.derive("replay/oracle"),
                pace_overrides=overrides,
            )
            replays.append(compare(oracle_trace, observed, pace_source="oracle"))

    return replays, frees


def validate_holdout(
    settings: AppSettings | None = None,
    params: ParameterSet | None = None,
    *,
    free_runs: int = DEFAULT_FREE_RUNS,
    anchors: Sequence[AnchorMode] = (AnchorMode.ANCHORED, AnchorMode.FITTED),
    oracle: bool = True,
    limit: int | None = None,
) -> ValidationReport:
    """Validate the twin on every dry race of the held-out seasons.

    Args:
        settings: Configuration; the process-wide settings by default.
        params: Fitted parameters; read from ``params/params.json`` by default.
        free_runs: Seeds per race in free mode. Zero skips free mode, which is much faster.
        anchors: Anchors to run under.
        oracle: Also run the leakage diagnostic that decomposes the error.
        limit: Validate only the first N races. For development; a published number uses them all.

    Raises:
        ValueError: if the parameters were fitted on a season being validated, which would make
            every number meaningless, or if no holdout race is ingested.
    """
    settings = settings or get_settings()
    params = params or ParameterSet.read(params_path(settings))

    holdout = settings.data.holdout_seasons
    contaminated = set(params.provenance.seasons) & set(holdout)
    if contaminated:
        raise ValueError(
            f"parameters were fitted on {sorted(contaminated)}, which is being validated; "
            "recalibrate on the training seasons before validating"
        )

    data = prepare_laps(holdout, settings)
    if data.all_laps.height == 0:
        raise ValueError(f"no ingested races for holdout seasons {list(holdout)}")

    simulator = RaceSimulator(params, settings)
    root = settings.seeds().derive("validation")

    races = data.all_laps.select("season", "round").unique().sort("season", "round").rows()
    if limit is not None:
        races = races[:limit]

    replays: list[RaceComparison] = []
    frees: list[FreeSummary] = []
    for season, round_number in races:
        laps = data.all_laps.filter(
            (pl.col("season") == season) & (pl.col("round") == round_number)
        )
        race_replays, race_frees = validate_race(
            simulator,
            laps,
            root.derive(f"{season}-{round_number}"),
            free_runs=free_runs,
            anchors=anchors,
            oracle=oracle,
        )
        replays.extend(race_replays)
        frees.extend(race_frees)

    return ValidationReport(
        seasons=tuple(holdout),
        n_races=len(races),
        replay=tuple(replays),
        free=tuple(frees),
        params_config_hash=params.provenance.config_hash,
        params_seasons=params.provenance.seasons,
    )
