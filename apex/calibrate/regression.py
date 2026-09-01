"""Least squares with absorbed fixed effects and cluster-robust standard errors.

Every car in every race has its own baseline pace -- the car, the driver, the circuit, the day.
None of that is a parameter the simulator wants, and all of it is correlated with the things it
does want. The standard remedy is a dummy per car-race, but that is roughly two thousand columns
over sixty thousand laps, so the equivalent within transformation is used instead: subtract each
car-race's mean from every variable and fit the remainder. By the Frisch-Waugh-Lovell theorem the
surviving coefficients are numerically identical to the dummy regression's, at a fraction of the
memory. The test suite asserts that identity against ``statsmodels`` rather than trusting it.

Standard errors are clustered by race. Laps within a race share a track temperature, a tyre
allocation and a set of incidents, so treating sixty thousand laps as sixty thousand independent
observations would produce intervals several times too narrow -- exactly the flattering
uncertainty ADR-0002 exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from apex.calibrate.estimates import Estimate

__all__ = ["RegressionResult", "fit_within_ols", "index_of", "within_transform"]

Vector = npt.NDArray[np.float64]
Matrix = npt.NDArray[np.float64]
Index = npt.NDArray[np.int64]

_RANK_TOLERANCE: Final = 1e-10
_MATRIX_DIMENSIONS: Final = 2
_MIN_CLUSTERS: Final = 2
"""Fewer than two clusters leaves nothing to estimate between-cluster variation from."""


def index_of(labels: npt.NDArray[np.str_] | list[str]) -> tuple[Index, int]:
    """Map labels to dense integer codes.

    Returns:
        The code per row, and the number of distinct labels.
    """
    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    return codes.astype(np.int64), int(codes.max()) + 1 if codes.size else 0


def within_transform(values: Matrix, groups: Index, n_groups: int) -> Matrix:
    """Subtract each group's mean from every column.

    Args:
        values: Observations by column, shape ``(n, k)``.
        groups: Dense group code per row.
        n_groups: Number of distinct groups.
    """
    if values.ndim != _MATRIX_DIMENSIONS:
        raise ValueError(f"expected a 2-D array, got shape {values.shape}")
    counts = np.bincount(groups, minlength=n_groups).astype(np.float64)
    counts[counts == 0.0] = 1.0
    means = np.stack(
        [
            np.bincount(groups, weights=values[:, column], minlength=n_groups) / counts
            for column in range(values.shape[1])
        ],
        axis=1,
    )
    return values - means[groups]


@dataclass(frozen=True, slots=True)
class RegressionResult:
    """Coefficients and their cluster-robust covariance.

    Attributes:
        names: Regressor names, in the order of ``coefficients``.
        coefficients: Point estimates.
        covariance: Cluster-robust covariance matrix.
        n_obs: Observations used.
        n_absorbed: Fixed effects absorbed by the within transformation.
        n_clusters: Clusters the standard errors are computed over.
        residual_std: Residual standard deviation, in the units of the outcome.
        r_squared_within: Share of within-group variance explained.
    """

    names: tuple[str, ...]
    coefficients: Vector
    covariance: Matrix
    n_obs: int
    n_absorbed: int
    n_clusters: int
    residual_std: float
    r_squared_within: float

    @property
    def std_errors(self) -> Vector:
        return np.sqrt(np.diag(self.covariance))

    @property
    def degrees_of_freedom(self) -> int:
        """Clusters minus one: with clustered errors, the clusters are the sample size."""
        return self.n_clusters - 1

    def estimate(self, name: str, level: float, *, n_obs: int | None = None) -> Estimate:
        """The named coefficient as a publishable :class:`Estimate`."""
        position = self.names.index(name)
        return Estimate.from_standard_error(
            float(self.coefficients[position]),
            float(self.std_errors[position]),
            n_obs=self.n_obs if n_obs is None else n_obs,
            level=level,
            degrees_of_freedom=float(self.degrees_of_freedom),
        )


def fit_within_ols(
    outcome: Vector,
    regressors: Matrix,
    names: tuple[str, ...],
    *,
    groups: Index,
    n_groups: int,
    clusters: Index,
    n_clusters: int,
) -> RegressionResult:
    """Fit ``outcome`` on ``regressors`` with group fixed effects absorbed.

    Args:
        outcome: Dependent variable, shape ``(n,)``.
        regressors: Design matrix without an intercept -- the fixed effects supply it.
        names: One name per column of ``regressors``.
        groups: Fixed-effect code per row, e.g. the car-race.
        n_groups: Number of distinct fixed effects.
        clusters: Cluster code per row, e.g. the race.
        n_clusters: Number of distinct clusters.

    Raises:
        ValueError: on shape mismatches, too few observations, or a rank-deficient design. A
            rank-deficient fit is refused rather than silently pseudo-inverted, because the
            coefficient it would return is not identified by the data.
    """
    n_obs, n_regressors = regressors.shape
    if outcome.shape != (n_obs,):
        raise ValueError(f"outcome has shape {outcome.shape}, expected ({n_obs},)")
    if len(names) != n_regressors:
        raise ValueError(f"{len(names)} names for {n_regressors} regressors")
    if n_clusters < _MIN_CLUSTERS:
        raise ValueError("cluster-robust errors need at least two clusters")

    residual_dof = n_obs - n_regressors - n_groups
    if residual_dof <= 0:
        raise ValueError(
            f"{n_obs} observations cannot support {n_regressors} regressors "
            f"and {n_groups} absorbed effects"
        )

    # A single missing value propagates through the normal equations and surfaces much later as
    # an opaque linear-algebra failure, so it is caught here where the column can still be named.
    for position, name in enumerate(names):
        if not np.isfinite(regressors[:, position]).all():
            raise ValueError(f"regressor {name!r} contains non-finite values")
    if not np.isfinite(outcome).all():
        raise ValueError("outcome contains non-finite values")

    demeaned_y = within_transform(outcome.reshape(-1, 1), groups, n_groups).ravel()
    demeaned_x = within_transform(regressors, groups, n_groups)

    gram = demeaned_x.T @ demeaned_x
    if np.linalg.matrix_rank(gram, tol=_RANK_TOLERANCE * max(1.0, float(np.trace(gram)))) < (
        n_regressors
    ):
        raise ValueError(f"design matrix is rank deficient in columns {names}")
    gram_inverse = np.linalg.inv(gram)
    coefficients = gram_inverse @ (demeaned_x.T @ demeaned_y)

    residuals = demeaned_y - demeaned_x @ coefficients
    scores = demeaned_x * residuals[:, None]
    # Sum the score vectors within each cluster, one regressor at a time, so the intermediate
    # never materialises as an (n_clusters, k, k) array.
    cluster_scores = np.stack(
        [
            np.bincount(clusters, weights=scores[:, column], minlength=n_clusters)
            for column in range(n_regressors)
        ],
        axis=1,
    )
    meat = cluster_scores.T @ cluster_scores

    # Stata's finite-sample correction for clustered errors with absorbed fixed effects.
    correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / residual_dof)
    covariance = gram_inverse @ meat @ gram_inverse * correction

    total_sum_of_squares = float(demeaned_y @ demeaned_y)
    residual_sum_of_squares = float(residuals @ residuals)
    return RegressionResult(
        names=names,
        coefficients=coefficients,
        covariance=covariance,
        n_obs=n_obs,
        n_absorbed=n_groups,
        n_clusters=n_clusters,
        residual_std=float(np.sqrt(residual_sum_of_squares / residual_dof)),
        r_squared_within=(
            1.0 - residual_sum_of_squares / total_sum_of_squares
            if total_sum_of_squares > 0.0
            else 0.0
        ),
    )
