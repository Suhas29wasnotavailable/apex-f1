"""Whether the car behind gets past, and how much harder that is at Monaco than at Monza.

P1 fitted nothing about overtaking, and a race simulator without it is not a race: cars would
change position only in the pit lane, a faster car stuck behind a slower one would stay there for
ninety laps, and the strategic question the whole project is about -- is it worth losing track
position for fresher tyres -- would have no cost on one side of the trade.

The measurement is a direct one. For every lap, take each pair of cars adjacent in the running
order, and ask whether the one behind was ahead a lap later. That is an observed on-track pass, and
its probability is modelled against the two things that plausibly drive it:

*   **The gap.** A car two seconds back cannot attempt a move; a car three tenths back can.
*   **The pace advantage.** A car half a second a lap faster will find a way past eventually; one
    a tenth faster mostly will not.

Fitted by logistic regression, with the intercept free to vary by circuit. The intercept is the
circuit's overtaking difficulty, and unlike the safety-car hazard it is richly identified: every
lap of every race contributes roughly twenty adjacent pairs, so even a circuit with one training
race brings thousands of observations rather than sixty. This is the one place in the project where
per-circuit estimates are strong enough to be interesting on their own.

**What is deliberately excluded, and why each exclusion matters.**

*Pit stops.* A car that pits drops several positions with nobody overtaking anybody. Including
those would inflate every pass rate and would do it most at circuits with the longest pit lanes,
which is exactly backwards.

*Neutralisations.* Overtaking is forbidden under a safety car, and the field bunches up, so those
laps combine near-zero gaps with zero passes. They would tell the model that a tiny gap predicts no
pass, inverting the effect being measured.

*Retirements and lapped traffic.* A car that stops is passed by everyone, and a car being lapped is
in a different contest from the one the running order implies. Both are dropped.

**The pace advantage is instantaneous, not seasonal, and that choice carries the undercut.** A
car's whole-race pace is the obvious thing to compare, and it is the wrong thing: the reason a
driver emerges from the pit lane and scythes past three cars is not that the car became better, it
is that the tyres are fifteen laps younger. So the advantage compared here is each car's pace *on
that lap* -- its car-race pace plus what its current compound and tyre age cost it, from
:func:`~apex.calibrate.baseline.tyre_delta`. Both terms are observed, and neither is contaminated
by the lap time of a car that was stuck in traffic.

That definition is also the one the simulator can actually evaluate. It knows every car's compound
and tyre age on every lap and does not know any car's whole-race average until the race is over, so
fitting on the seasonal quantity would have left the simulator substituting a correlated but
different number into a coefficient fitted for something else. Calibration and simulation compute
this from the same function for exactly that reason.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.calibrate.baseline import tyre_delta
from apex.calibrate.estimates import Estimate
from apex.calibrate.exclusions import is_safety_car_lap
from apex.calibrate.pace import PaceModel
from apex.calibrate.pooling import partial_pool
from apex.config import AppSettings, get_settings

__all__ = ["CircuitOvertaking", "OvertakingModel", "fit_overtaking", "passing_attempts"]

_MAX_GAP_S = 5.0
"""Beyond this the car behind is not racing the car ahead in any meaningful sense, and including
those pairs would let a mass of certain non-passes drown the informative ones."""

_MIN_PAIRS_PER_CIRCUIT = 200
"""Below this a circuit takes the pooled difficulty rather than its own."""

_NEWTON_STEPS = 50
_CONVERGENCE = 1e-10


class CircuitOvertaking(BaseModel):
    """How hard it is to pass at one circuit, or pooled across all of them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit: str | None = None
    difficulty: Estimate
    """Logistic intercept: the log-odds of a pass at zero gap and zero pace advantage. More
    negative is harder to pass at."""

    n_pairs: int = Field(ge=0)
    n_passes: int = Field(ge=0)
    n_races: int = Field(ge=0)

    @property
    def pass_rate(self) -> float:
        """Raw share of adjacent pairs that resulted in a pass, before any modelling."""
        return self.n_passes / self.n_pairs if self.n_pairs else 0.0


class OvertakingModel(BaseModel):
    """The fitted passing model: one shared shape, one difficulty per circuit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pooled: CircuitOvertaking
    circuits: tuple[CircuitOvertaking, ...]

    gap_coefficient: Estimate
    """Change in log-odds of a pass per second of gap to the car ahead. Expected negative: the
    further back, the less chance."""

    pace_coefficient: Estimate
    """Change in log-odds per second per lap of pace advantage. Expected positive: a faster car
    passes more."""

    between_circuit_std: float
    """Spread of difficulty across circuits. Unlike the safety-car hazard this is expected to be
    clearly non-zero, because the difference between a street circuit and a power track is real and
    there is enough data per circuit to see it."""

    max_gap_s: float
    """The window pairs were drawn from, echoed so the coefficients cannot be read without it."""

    n_pairs: int = Field(ge=0)
    n_passes: int = Field(ge=0)
    n_races: int = Field(ge=0)

    def circuit(self, name: str) -> CircuitOvertaking:
        for entry in self.circuits:
            if entry.circuit == name:
                return entry
        raise KeyError(f"no overtaking model fitted for circuit {name!r}")

    def difficulty_at(self, circuit: str | None) -> float:
        """Difficulty at a circuit, falling back to the pooled value for an unknown one."""
        if circuit is not None:
            for entry in self.circuits:
                if entry.circuit == circuit:
                    return entry.difficulty.value
        return self.pooled.difficulty.value

    def pass_probability(
        self, gap_s: float, pace_advantage_s: float, circuit: str | None = None
    ) -> float:
        """Probability the car behind completes a pass over one lap.

        Args:
            gap_s: Gap to the car ahead at the start/finish line, in seconds.
            pace_advantage_s: How much faster the car behind is, in seconds per lap. Positive
                favours the attacker.
            circuit: Circuit name; the pooled difficulty is used when it is unknown.

        Returns:
            Zero beyond the fitted gap window rather than an extrapolated tail. The model saw no
            pairs out there and should not pretend to know what happens.
        """
        if gap_s > self.max_gap_s:
            return 0.0
        log_odds = (
            self.difficulty_at(circuit)
            + self.gap_coefficient.value * max(gap_s, 0.0)
            + self.pace_coefficient.value * pace_advantage_s
        )
        return float(1.0 / (1.0 + np.exp(-log_odds)))


def passing_attempts(
    all_laps: pl.DataFrame,
    cars: pl.DataFrame,
    pace: PaceModel | Mapping[str, PaceModel],
    settings: AppSettings | None = None,
) -> pl.DataFrame:
    """Every pair of adjacent cars that had a clean racing lap, and whether the pass happened.

    Args:
        all_laps: Featured, *unfiltered* laps. The unfiltered frame is required: the running order
            has to include cars on laps the pace filter removes, or the car ahead goes missing and
            its follower is paired with whoever survived instead.
        cars: Car-race pace, from :func:`~apex.calibrate.baseline.car_pace_frame`, supplying each
            car's pace so the advantage between a pair can be computed.
        pace: The fitted pace model, pooled or per circuit, supplying the tyre term that turns a
            car's whole-race pace into its pace on this particular lap.
        settings: Configuration; the process-wide settings by default.

    Returns:
        One row per adjacent pair per lap, with the circuit, the gap, the tyre-adjusted pace
        advantage of the car behind, and whether it was ahead a lap later.
    """
    settings = settings or get_settings()

    pace_by_car = cars.select(
        pl.col("race_id"),
        pl.col("driver"),
        pl.col("pace_s"),
    )
    joined = all_laps.join(pace_by_car, on=["race_id", "driver"], how="inner")
    if joined.height == 0:
        return pl.DataFrame(schema=_ATTEMPT_SCHEMA)

    # Each car's pace on this lap: what the car is worth over a race, plus what its current tyre
    # costs it right now. This is the quantity the simulator evaluates, computed by the same code.
    laps = (
        _with_lap_pace(joined, pace)
        .with_columns(is_safety_car_lap(settings).alias("_neutralised"))
        .filter(pl.col("position").is_not_null() & pl.col("lap_time_s").is_not_null())
    )
    if laps.height == 0:
        return pl.DataFrame(schema=_ATTEMPT_SCHEMA)

    # Next lap's position for the same car, and whether that car pitted on either lap.
    keys = ["race_id", "driver"]
    ordered = laps.sort(*keys, "lap").with_columns(
        pl.col("position").shift(-1).over(keys).alias("_next_position"),
        pl.col("lap").shift(-1).over(keys).alias("_next_lap"),
        (pl.col("is_pit_in") | pl.col("is_pit_out")).alias("_pit"),
        (pl.col("is_pit_in") | pl.col("is_pit_out")).shift(-1).over(keys).alias("_next_pit"),
        pl.col("_neutralised").shift(-1).over(keys).alias("_next_neutralised"),
    )

    # A car only counts if it is still running next lap, and that lap is the very next one: a gap
    # in the sequence means a retirement or a missing lap, not a completed racing lap.
    racing = ordered.filter(
        pl.col("_next_position").is_not_null()
        & (pl.col("_next_lap") == pl.col("lap") + 1)
        & ~pl.col("_pit")
        & ~pl.col("_next_pit").fill_null(value=True)
        & ~pl.col("_neutralised")
        & ~pl.col("_next_neutralised").fill_null(value=True)
    ).select(
        "race_id",
        "circuit",
        "lap",
        "driver",
        "position",
        "_next_position",
        "gap_ahead_s",
        "lap_pace_s",
    )

    # Pair each car with the car directly ahead of it in the running order on the same lap. The
    # join on position + 1 is what makes the pair adjacent; a non-adjacent pair is a different
    # question and is not a pass when the order changes.
    ahead = racing.select(
        pl.col("race_id"),
        pl.col("lap"),
        pl.col("position").alias("_ahead_position"),
        pl.col("driver").alias("ahead_driver"),
        pl.col("_next_position").alias("_ahead_next_position"),
        pl.col("lap_pace_s").alias("_ahead_lap_pace_s"),
    )
    pairs = racing.join(
        ahead,
        left_on=["race_id", "lap", pl.col("position") - 1],
        right_on=["race_id", "lap", "_ahead_position"],
        how="inner",
    )

    return (
        pairs.with_columns(
            (pl.col("_ahead_lap_pace_s") - pl.col("lap_pace_s")).alias("pace_advantage_s"),
            (pl.col("_next_position") < pl.col("_ahead_next_position")).alias("passed"),
            pl.col("gap_ahead_s").fill_null(_MAX_GAP_S + 1.0).alias("gap_s"),
        )
        .filter(pl.col("gap_s") <= _MAX_GAP_S)
        .select(
            "race_id",
            "circuit",
            "lap",
            pl.col("driver").alias("behind_driver"),
            "ahead_driver",
            "gap_s",
            "pace_advantage_s",
            "passed",
        )
        .sort("race_id", "lap", "behind_driver")
    )


def _with_lap_pace(laps: pl.DataFrame, pace: PaceModel | Mapping[str, PaceModel]) -> pl.DataFrame:
    """Attach each lap's tyre-adjusted pace, using the per-circuit model where one exists."""
    if isinstance(pace, PaceModel):
        return laps.with_columns(
            (pl.col("pace_s") + tyre_delta(pace)).alias("lap_pace_s")
        ).drop_nulls("lap_pace_s")

    frames = [
        part.with_columns((pl.col("pace_s") + tyre_delta(pace[str(circuit)])).alias("lap_pace_s"))
        for (circuit,), part in laps.group_by(["circuit"], maintain_order=True)
        if str(circuit) in pace
    ]
    if not frames:
        raise ValueError("no circuit in the laps has a fitted pace model")
    return pl.concat(frames).drop_nulls("lap_pace_s")


_ATTEMPT_SCHEMA: dict[str, pl.DataType] = {
    "race_id": pl.String(),
    "circuit": pl.String(),
    "lap": pl.Int16(),
    "behind_driver": pl.String(),
    "ahead_driver": pl.String(),
    "gap_s": pl.Float64(),
    "pace_advantage_s": pl.Float64(),
    "passed": pl.Boolean(),
}


def _logistic_fit(
    design: np.ndarray, outcome: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood logistic regression with cluster-robust standard errors.

    Newton-Raphson, which converges in a handful of steps on a problem this well conditioned.
    Standard errors are clustered by race for the same reason as everywhere else in this project:
    the pairs within one race share a safety car, a track temperature and a tyre allocation, so
    treating twenty thousand of them as independent would produce intervals far too narrow.
    """
    beta = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(_NEWTON_STEPS):
        probability = 1.0 / (1.0 + np.exp(-design @ beta))
        weight = np.clip(probability * (1.0 - probability), 1e-12, None)
        gradient = design.T @ (outcome - probability)
        hessian = (design * weight[:, None]).T @ design
        step = np.linalg.solve(hessian + np.eye(design.shape[1]) * 1e-10, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < _CONVERGENCE:
            break

    probability = 1.0 / (1.0 + np.exp(-design @ beta))
    weight = np.clip(probability * (1.0 - probability), 1e-12, None)
    bread = np.linalg.pinv((design * weight[:, None]).T @ design)
    score = design * (outcome - probability)[:, None]

    meat = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    order = np.argsort(clusters, kind="stable")
    labels = clusters[order]
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for group in np.split(order, boundaries):
        total = score[group].sum(axis=0)
        meat += np.outer(total, total)

    n_clusters = len(np.unique(clusters))
    correction = n_clusters / max(n_clusters - 1, 1) if n_clusters > 1 else 1.0
    covariance = bread @ meat @ bread * correction
    return beta, np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def fit_overtaking(
    all_laps: pl.DataFrame,
    cars: pl.DataFrame,
    pace: PaceModel | Mapping[str, PaceModel],
    settings: AppSettings | None = None,
) -> OvertakingModel:
    """Fit pass probability from observed position changes.

    Args:
        all_laps: Featured, unfiltered laps -- ``CalibrationData.all_laps``.
        cars: Car-race pace, from :func:`~apex.calibrate.baseline.car_pace_frame`.
        pace: The fitted pace model, pooled or per circuit, supplying the tyre term.
        settings: Configuration; the process-wide settings by default.

    Raises:
        ValueError: if no clean racing pair could be measured.
    """
    settings = settings or get_settings()
    level = settings.calibration.confidence_level

    attempts = passing_attempts(all_laps, cars, pace, settings)
    if attempts.height == 0:
        raise ValueError("no adjacent racing pairs found; cannot fit an overtaking model")

    gap = attempts["gap_s"].to_numpy().astype(np.float64)
    advantage = attempts["pace_advantage_s"].to_numpy().astype(np.float64)
    outcome = attempts["passed"].to_numpy().astype(np.float64)
    races = attempts["race_id"].to_numpy()

    # The shared shape is fitted once on every pair, with a single intercept. Circuit difficulty is
    # then measured as a per-circuit intercept holding that shape fixed, and shrunk. Fitting one
    # slope per circuit as well would spend the data on twenty-five copies of the same curve.
    design = np.column_stack([np.ones_like(gap), gap, advantage])
    beta, errors = _logistic_fit(design, outcome, races)

    n_races = int(attempts.select(pl.n_unique("race_id")).item())
    pooled = CircuitOvertaking(
        circuit=None,
        difficulty=Estimate.from_standard_error(
            float(beta[0]),
            float(errors[0]),
            n_obs=attempts.height,
            level=level,
            degrees_of_freedom=float(max(n_races - 1, 1)),
        ),
        n_pairs=attempts.height,
        n_passes=int(attempts["passed"].sum()),
        n_races=n_races,
    )

    offset = beta[1] * gap + beta[2] * advantage
    local: dict[str, Estimate] = {}
    counts: dict[str, tuple[int, int, int]] = {}
    for (circuit,), part in attempts.group_by(["circuit"], maintain_order=True):
        name = str(circuit)
        mask = attempts["circuit"].to_numpy() == name
        n_pairs = int(mask.sum())
        counts[name] = (
            n_pairs,
            int(part["passed"].sum()),
            int(part.select(pl.n_unique("race_id")).item()),
        )
        if n_pairs < _MIN_PAIRS_PER_CIRCUIT:
            continue
        # The intercept is refitted with the shared slopes held as a known offset, which is what
        # makes this circuit's number comparable with the pooled one.
        local[name] = _intercept_with_offset(
            outcome[mask], offset[mask], races[mask], level, counts[name][2]
        )

    if not local:
        circuits = tuple(
            CircuitOvertaking(
                circuit=name,
                difficulty=pooled.difficulty.model_copy(update={"shrinkage": 0.0}),
                n_pairs=n_pairs,
                n_passes=n_passes,
                n_races=circuit_races,
            )
            for name, (n_pairs, n_passes, circuit_races) in sorted(counts.items())
        )
        between_std = 0.0
    else:
        pooling = partial_pool(local, level=level, toward=pooled.difficulty)
        between_std = pooling.between_std
        circuits = tuple(
            CircuitOvertaking(
                circuit=name,
                difficulty=pooling.estimates.get(
                    name, pooled.difficulty.model_copy(update={"shrinkage": 0.0})
                ),
                n_pairs=n_pairs,
                n_passes=n_passes,
                n_races=circuit_races,
            )
            for name, (n_pairs, n_passes, circuit_races) in sorted(counts.items())
        )

    return OvertakingModel(
        pooled=pooled,
        circuits=circuits,
        gap_coefficient=Estimate.from_standard_error(
            float(beta[1]),
            float(errors[1]),
            n_obs=attempts.height,
            level=level,
            degrees_of_freedom=float(max(n_races - 1, 1)),
        ),
        pace_coefficient=Estimate.from_standard_error(
            float(beta[2]),
            float(errors[2]),
            n_obs=attempts.height,
            level=level,
            degrees_of_freedom=float(max(n_races - 1, 1)),
        ),
        between_circuit_std=between_std,
        max_gap_s=_MAX_GAP_S,
        n_pairs=attempts.height,
        n_passes=int(attempts["passed"].sum()),
        n_races=n_races,
    )


def _intercept_with_offset(
    outcome: np.ndarray, offset: np.ndarray, clusters: np.ndarray, level: float, n_races: int
) -> Estimate:
    """Fit an intercept alone, with the shared slopes' contribution held fixed.

    This is what makes a circuit's difficulty directly comparable with the pooled intercept: both
    are the log-odds of a pass once gap and pace advantage are accounted for by the same curve.
    """
    beta = np.zeros(1, dtype=np.float64)
    for _ in range(_NEWTON_STEPS):
        probability = 1.0 / (1.0 + np.exp(-(beta[0] + offset)))
        weight = np.clip(probability * (1.0 - probability), 1e-12, None)
        gradient = float((outcome - probability).sum())
        hessian = float(weight.sum())
        step = gradient / max(hessian, 1e-12)
        beta[0] += step
        if abs(step) < _CONVERGENCE:
            break

    probability = 1.0 / (1.0 + np.exp(-(beta[0] + offset)))
    weight = np.clip(probability * (1.0 - probability), 1e-12, None)
    bread = 1.0 / max(float(weight.sum()), 1e-12)
    residual = outcome - probability

    meat = 0.0
    order = np.argsort(clusters, kind="stable")
    labels = clusters[order]
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for group in np.split(order, boundaries):
        meat += float(residual[group].sum()) ** 2

    n_clusters = len(np.unique(clusters))
    correction = n_clusters / max(n_clusters - 1, 1) if n_clusters > 1 else 1.0
    std_error = float(np.sqrt(bread * meat * bread * correction))
    return Estimate.from_standard_error(
        float(beta[0]),
        std_error,
        n_obs=int(outcome.size),
        level=level,
        degrees_of_freedom=float(max(n_races - 1, 1)),
    )
