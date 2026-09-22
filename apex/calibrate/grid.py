"""What a starting grid says about race pace.

The simulator has to be able to build a field from information that exists *before* the race, for
two independent reasons.

**Validation would otherwise be circular.** A held-out 2025 race can only test the twin if the
twin was not told the answer. Per-car pace read from the race's own lap times is the largest single
determinant of who finishes where, so handing it over would leave almost nothing for the simulator
to get wrong. Grid position is known before the lights go out, so a field built from it is a
prediction rather than a restatement.

**A hypothetical race has no lap times at all.** The user-facing promise is to enter a lineup and
watch the race happen. There are no telemetry-derived pace figures for a grid somebody typed in,
so the mapping from starting position to race pace has to be a fitted model in its own right.

Grid position is a good proxy and an imperfect one, which is the honest description. It is set by a
single qualifying lap on low fuel and new tyres, while race pace is a sixty-lap average on a heavy
car; a team strong over one lap and weak over a stint sits in the residual. That residual is fitted
and published rather than assumed away, because it is what makes a simulated field spread
realistically instead of collapsing onto a smooth curve.

The fitted shape is quadratic in normalised grid position. The data is visibly concave -- the gap
from pole to second is larger than from fifteenth to sixteenth, because the front of the grid is
where the fastest cars are densely packed and the back is a plateau -- and a quadratic captures
that with two coefficients where a per-position mean would spend twenty on reproducing noise.

Pit-lane starts are excluded from the fit and only from the fit. FastF1 reports them as grid
position zero, and they are not the twenty-first-fastest car: a front-running car sent to the pit
lane for a technical infringement starts last and races like a front-runner. Including them would
tell the model that the very back of the grid is unexpectedly quick. They still start from the
back, which is a fact about track position, not about pace.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.calibrate.estimates import Estimate
from apex.config import AppSettings, get_settings

__all__ = ["PIT_LANE_GRID", "GridPaceModel", "fit_grid_pace"]

PIT_LANE_GRID = 0
"""FastF1's grid position for a car that started from the pit lane."""

_REFERENCE_FIELD_SIZE = 20
"""Grid size the normalisation is defined against, so a coefficient means the same thing in a race
with a depleted grid as in a full one."""

_POLYNOMIAL_DEGREE = 2


class GridPaceModel(BaseModel):
    """Expected race pace as a function of starting position."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coefficients: tuple[Estimate, ...] = Field(min_length=1)
    """Polynomial in normalised grid position, constant term first. Normalisation is
    ``(grid - 1) / (reference_field_size - 1)``, so the argument runs from 0 on pole to 1 at the
    back of a full grid."""

    reference_field_size: int = Field(ge=2)

    residual_std_s: Estimate
    """Spread of car pace around the fitted curve. The part of race pace that qualifying does not
    predict -- a car better over a stint than over a lap -- and the amount of realistic disorder a
    constructed field needs."""

    n_car_races: int = Field(ge=0)
    n_races: int = Field(ge=0)
    r_squared: float
    """Share of car-race pace variance explained by grid position alone. Expected to be well
    under one; were it near one, qualifying would decide races and there would be little left for a
    simulator to determine."""

    def normalised(self, grid_position: int) -> float:
        """Map a grid position onto the fitted curve's argument."""
        return (grid_position - 1) / (self.reference_field_size - 1)

    def pace_s(self, grid_position: int) -> float:
        """Expected pace in seconds per lap relative to the field median. Negative is faster.

        A pit-lane start is evaluated at the back of the reference grid, which is the only defined
        answer: its true pace is unrelated to its starting position, so a caller who knows better
        should override it rather than trust this.
        """
        position = self.reference_field_size if grid_position == PIT_LANE_GRID else grid_position
        argument = self.normalised(position)
        return float(
            sum(term.value * argument**degree for degree, term in enumerate(self.coefficients))
        )


def fit_grid_pace(cars: pl.DataFrame, settings: AppSettings | None = None) -> GridPaceModel:
    """Fit race pace against starting position.

    Args:
        cars: Car-race pace, from :func:`~apex.calibrate.baseline.car_pace_frame`.
        settings: Configuration; the process-wide settings by default.

    Raises:
        ValueError: if no car-race has a usable grid position.
    """
    settings = settings or get_settings()
    level = settings.calibration.confidence_level

    usable = cars.filter(pl.col("grid_position") > PIT_LANE_GRID)
    if usable.height <= _POLYNOMIAL_DEGREE:
        raise ValueError(
            f"need more than {_POLYNOMIAL_DEGREE} car-races with a grid position to fit, "
            f"got {usable.height}"
        )

    argument = (usable["grid_position"].to_numpy().astype(np.float64) - 1.0) / (
        _REFERENCE_FIELD_SIZE - 1
    )
    target = usable["pace_s"].to_numpy().astype(np.float64)
    design = np.vander(argument, _POLYNOMIAL_DEGREE + 1, increasing=True)

    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ solution
    dof = max(usable.height - solution.size, 1)

    # Standard errors clustered by race. Two cars of the same team in the same race share most of
    # what makes them fast, so treating forty cars in a race weekend as forty independent
    # observations would understate every interval here by roughly the square root of the team size.
    races = usable["race_id"].to_list()
    covariance = _clustered_covariance(design, residuals, races)
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    n_races = len({*races})

    total_variance = float(((target - target.mean()) ** 2).sum())
    r_squared = (
        0.0 if total_variance <= 0.0 else 1.0 - float(residuals @ residuals) / total_variance
    )

    return GridPaceModel(
        coefficients=tuple(
            Estimate.from_standard_error(
                float(solution[degree]),
                float(errors[degree]),
                n_obs=usable.height,
                level=level,
                degrees_of_freedom=float(max(n_races - 1, 1)),
            )
            for degree in range(solution.size)
        ),
        reference_field_size=_REFERENCE_FIELD_SIZE,
        residual_std_s=Estimate.from_standard_error(
            float(np.sqrt(float(residuals @ residuals) / dof)),
            # A standard deviation's own standard error, under normality.
            float(np.sqrt(float(residuals @ residuals) / dof) / np.sqrt(2.0 * dof)),
            n_obs=usable.height,
            level=level,
            degrees_of_freedom=float(dof),
        ),
        n_car_races=usable.height,
        n_races=n_races,
        r_squared=r_squared,
    )


def _clustered_covariance(
    design: np.ndarray, residuals: np.ndarray, clusters: list[str]
) -> np.ndarray:
    """Cluster-robust covariance of a least-squares fit, grouping by race."""
    bread = np.linalg.pinv(design.T @ design)
    meat = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    order = np.argsort(clusters, kind="stable")
    labels = np.asarray(clusters)[order]
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for group in np.split(order, boundaries):
        score = design[group].T @ residuals[group]
        meat += np.outer(score, score)

    n_clusters = len(np.unique(labels))
    n_obs, n_terms = design.shape
    correction = (
        (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / max(n_obs - n_terms, 1))
        if n_clusters > 1
        else 1.0
    )
    return bread @ meat @ bread * correction
