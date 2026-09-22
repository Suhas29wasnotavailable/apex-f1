"""How a simulated race is scored against the real one.

The metrics are chosen so that a good score is hard to get for the wrong reason.

**Position error, not just the winner.** Predicting the winner is nearly free: the fastest car
usually starts at the front and stays there, so a model that never moves anybody off the grid gets
a respectable winner accuracy. Mean absolute position error across the whole field is much harder
to fake, and the grid itself is reported alongside as the baseline every metric has to beat. A twin
whose
position error is no better than "everyone finishes where they started" has demonstrated nothing,
however good its other numbers look.

**Lap-time error separately from race-time error.** They fail differently. Lap-time error says
whether the pace model is right; total race time can be right while every lap is wrong, if the
errors cancel. Both are reported, matched driver by driver and lap by lap.

**Spearman rather than Pearson on the order.** The finishing order is a ranking, and a rank
correlation is the coefficient that treats it as one.

Every metric is computed only over drivers present in both races, and the count is reported. The
simulator models no retirements, so a real race with five of them is being compared on fifteen cars,
and a reader is entitled to know that before reading the error.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.eval.observed import ObservedRace
from apex.sim.trace import RaceTrace

__all__ = ["RaceComparison", "compare"]

_PODIUM = 3
_MIN_FOR_CORRELATION = 2
"""A rank correlation needs two points before it means anything."""


class RaceComparison(BaseModel):
    """How closely one simulated race matched the real one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: str
    season: int
    circuit: str
    anchor_mode: str
    pace_source: str = "grid"
    """Where each car's pace came from. ``grid`` is the fitted pre-race forecast and is the only
    value a published result may carry. ``oracle`` means the pace was read out of the race being
    predicted, which is leakage and is reported only to decompose the error."""

    n_compared: int = Field(ge=0)
    n_retired_excluded: int = Field(ge=0)

    position_mae: float
    """Mean absolute finishing-position error over drivers in both races."""

    grid_position_mae: float
    """The same error for a model that predicts everybody finishes where they started. The number
    every other metric here has to beat to mean anything."""

    order_spearman: float
    """Rank correlation between simulated and real finishing order."""

    winner_correct: bool
    podium_overlap: int = Field(ge=0, le=_PODIUM)
    top10_overlap: int = Field(ge=0, le=10)

    lap_time_mae_s: float | None
    """Mean absolute error of per-lap times, matched by driver and lap. ``None`` in a mode where lap
    times are not comparable."""

    winner_race_time_error_s: float | None
    """Signed error in the winner's total race time. Meaningful only when the pace level was
    anchored
    to this race; under a fitted anchor it is dominated by the season extrapolation."""

    finish_gap_mae_s: float | None
    """Mean absolute error of each car's final gap to the winner. A measure of whether the field
    spread out the way it really did, which position error alone cannot see."""

    n_timed: int = Field(default=0, ge=0)
    """Cars the time-based metrics covered: those finishing on the lead lap with every lap timed.
    Fewer than :attr:`n_compared` whenever a car was lapped, and reported so a gap error cannot be
    read without knowing how much of the field it describes."""

    @property
    def beats_grid_baseline(self) -> bool:
        """Whether the twin ordered the field better than the starting grid did."""
        return self.position_mae < self.grid_position_mae


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Rank correlation, computed directly so the metric does not depend on an optional extra."""
    if left.size < _MIN_FOR_CORRELATION:
        return float("nan")

    def ranks(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        result = np.empty_like(order, dtype=np.float64)
        result[order] = np.arange(values.size, dtype=np.float64)
        return result

    a, b = ranks(left), ranks(right)
    a_centred, b_centred = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a_centred**2).sum() * (b_centred**2).sum()))
    return float(a_centred @ b_centred / denominator) if denominator > 0.0 else float("nan")


def _lap_time_error(trace: RaceTrace, observed: ObservedRace, drivers: set[str]) -> float | None:
    """Mean absolute per-lap time error, matched by driver and lap.

    Pit laps and neutralised laps are included deliberately. They are part of the race, the
    simulator claims to model both, and excluding them would quietly drop the laps where the model
    is doing the most interesting work.
    """
    simulated = trace.frame().select("driver", "lap", pl.col("lap_time_s").alias("sim_s"))
    real = observed.laps.select("driver", "lap", pl.col("lap_time_s").alias("real_s"))
    matched = (
        simulated.join(real, on=["driver", "lap"], how="inner")
        .filter(pl.col("driver").is_in(list(drivers)))
        .drop_nulls(["sim_s", "real_s"])
    )
    if matched.height == 0:
        return None
    return float((matched["sim_s"] - matched["real_s"]).abs().mean())  # type: ignore[arg-type]


def _finish_gap_error(trace: RaceTrace, observed: ObservedRace, drivers: set[str]) -> float | None:
    """Mean absolute error in each car's final gap to the winner."""
    real_times = {
        driver: time for driver, time in observed.total_time_s.items() if driver in drivers
    }
    if len(real_times) < _MIN_FOR_CORRELATION:
        return None
    real_winner = min(real_times.values())
    simulated = {
        entry.driver: entry.total_time_s for entry in trace.results if entry.driver in drivers
    }
    if len(simulated) < _MIN_FOR_CORRELATION:
        return None
    sim_winner = min(simulated.values())
    errors = [
        abs((simulated[driver] - sim_winner) - (real_times[driver] - real_winner))
        for driver in real_times
        if driver in simulated
    ]
    return float(np.mean(errors)) if errors else None


def compare(
    trace: RaceTrace, observed: ObservedRace, *, pace_source: str = "grid"
) -> RaceComparison:
    """Score a simulated race against the real one it was built from.

    Args:
        trace: The simulated race.
        observed: The real race, reduced by :func:`~apex.eval.observed.observed_race`.
        pace_source: Label recording where the field's pace came from, carried into the result so an
            oracle run can never be mistaken for a prediction.

    Raises:
        ValueError: if no driver appears in both races, which means the two are not the same event.
    """
    simulated_order = list(trace.finishing_order())
    real_order = list(observed.finishing_order)
    drivers = set(simulated_order) & set(real_order)
    if not drivers:
        raise ValueError(
            f"no driver appears in both the simulated and real {observed.event}; "
            "the scenario and the race do not match"
        )

    sim_position = {driver: index + 1 for index, driver in enumerate(simulated_order)}
    real_position = {driver: index + 1 for index, driver in enumerate(real_order)}
    grid_position = {entry.driver: entry.grid_position for entry in trace.results}

    shared = sorted(drivers, key=lambda name: real_position[name])
    simulated_ranks = np.array([sim_position[name] for name in shared], dtype=np.float64)
    real_ranks = np.array([real_position[name] for name in shared], dtype=np.float64)

    # The grid baseline is re-ranked over the same drivers, so a race with retirements does not give
    # the baseline credit for slots nobody finished in.
    grid_order = sorted(shared, key=lambda name: (grid_position.get(name, 99), name))
    grid_ranks = np.array([grid_order.index(name) + 1 for name in shared], dtype=np.float64)

    real_top = [name for name in real_order if name in drivers]
    sim_top = [name for name in simulated_order if name in drivers]

    winner_time_error: float | None = None
    if real_top and sim_top:
        # ``total_time_s`` holds only lead-lap finishers, so a winner absent from it had an untimed
        # lap and no race time can be quoted for the comparison.
        real_winner_time = observed.total_time_s.get(real_top[0])
        simulated_winner = next(
            (entry for entry in trace.results if entry.driver == sim_top[0]), None
        )
        if real_winner_time is not None and simulated_winner is not None:
            winner_time_error = simulated_winner.total_time_s - real_winner_time

    return RaceComparison(
        event=observed.event,
        season=observed.season,
        circuit=observed.circuit,
        anchor_mode=str(trace.scenario.anchor.mode),
        pace_source=pace_source,
        n_compared=len(shared),
        n_retired_excluded=len(observed.retired),
        position_mae=float(np.abs(simulated_ranks - real_ranks).mean()),
        grid_position_mae=float(np.abs(grid_ranks - real_ranks).mean()),
        order_spearman=_spearman(simulated_ranks, real_ranks),
        winner_correct=bool(sim_top[0] == real_top[0]),
        podium_overlap=len(set(sim_top[:_PODIUM]) & set(real_top[:_PODIUM])),
        top10_overlap=len(set(sim_top[:10]) & set(real_top[:10])),
        lap_time_mae_s=_lap_time_error(trace, observed, drivers),
        winner_race_time_error_s=winner_time_error,
        finish_gap_mae_s=_finish_gap_error(trace, observed, drivers),
        n_timed=len(drivers & set(observed.total_time_s)),
    )
