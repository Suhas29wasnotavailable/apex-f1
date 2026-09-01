"""The estimator behind every fitted parameter, checked against a reference implementation.

The within transformation is an optimisation: it claims to give exactly the coefficients a
regression with one dummy per car-race would give, without building that matrix. A claim like
that is worth nothing unasserted, so it is asserted here against ``statsmodels`` -- coefficients,
cluster-robust standard errors and residual scale alike.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import statsmodels.api as sm

from apex.calibrate.regression import (
    RegressionResult,
    fit_within_ols,
    index_of,
    within_transform,
)

Matrix = npt.NDArray[np.float64]


def synthetic(
    n_groups: int = 12, per_group: int = 40, seed: int = 11
) -> tuple[Matrix, Matrix, list[str], list[str]]:
    """Panel data with a real group effect, so absorbing it actually matters."""
    rng = np.random.default_rng(seed)
    group_labels = [f"g{index}" for index in range(n_groups) for _ in range(per_group)]
    cluster_labels = [f"c{index // 3}" for index in range(n_groups) for _ in range(per_group)]
    n_obs = n_groups * per_group

    regressors = np.column_stack(
        [rng.normal(size=n_obs), rng.uniform(0.0, 1.0, size=n_obs), rng.normal(size=n_obs) ** 2]
    )
    effects = np.repeat(rng.normal(scale=5.0, size=n_groups), per_group)
    outcome = (
        90.0 + effects + regressors @ np.array([0.3, -1.2, 0.05]) + rng.normal(0.2, size=n_obs)
    )
    return outcome, regressors, group_labels, cluster_labels


def reference(
    outcome: Matrix, regressors: Matrix, group_labels: list[str], cluster_labels: list[str]
) -> tuple[Matrix, Matrix, float]:
    """The same model with explicit dummies, fitted by statsmodels.

    Only the numbers cross the boundary: statsmodels ships no type information, and letting its
    result object into a signature would put an untyped value into the project's own API surface.
    """
    codes, n_groups = index_of(group_labels)
    dummies = np.zeros((outcome.size, n_groups))
    dummies[np.arange(outcome.size), codes] = 1.0
    design = np.column_stack([regressors, dummies])
    cluster_codes, _ = index_of(cluster_labels)
    fit = sm.OLS(outcome, design).fit(cov_type="cluster", cov_kwds={"groups": cluster_codes})
    width = regressors.shape[1]
    return (
        np.asarray(fit.params[:width], dtype=np.float64),
        np.asarray(fit.bse[:width], dtype=np.float64),
        float(np.sqrt(fit.mse_resid)),
    )


def fitted() -> tuple[RegressionResult, tuple[Matrix, Matrix, float]]:
    outcome, regressors, group_labels, cluster_labels = synthetic()
    groups, n_groups = index_of(group_labels)
    clusters, n_clusters = index_of(cluster_labels)
    result = fit_within_ols(
        outcome,
        regressors,
        ("a", "b", "c"),
        groups=groups,
        n_groups=n_groups,
        clusters=clusters,
        n_clusters=n_clusters,
    )
    return result, reference(outcome, regressors, group_labels, cluster_labels)


def test_absorbing_reproduces_the_dummy_regression_coefficients() -> None:
    result, (coefficients, _, _) = fitted()
    assert result.coefficients == pytest.approx(coefficients, rel=1e-10)


def test_clustered_standard_errors_match_the_reference() -> None:
    result, (_, std_errors, _) = fitted()
    assert result.std_errors == pytest.approx(std_errors, rel=1e-8)


def test_residual_scale_matches_the_reference() -> None:
    result, (_, _, residual_std) = fitted()
    assert result.residual_std == pytest.approx(residual_std, rel=1e-10)


def test_clustered_errors_are_wider_when_the_effect_varies_between_clusters() -> None:
    """The reason for clustering by race, made concrete.

    Degradation is not identical from race to race: track temperature, tyre allocation and
    surface all move it. Under that -- a slope that varies by cluster -- treating laps as
    independent understates the uncertainty in the average slope by a wide margin, and the
    interval a reader is shown is the one that matters.
    """
    rng = np.random.default_rng(3)
    n_clusters, groups_per_cluster, per_group = 10, 3, 40
    n_obs = n_clusters * groups_per_cluster * per_group

    cluster_labels = [
        f"c{index}" for index in range(n_clusters) for _ in range(groups_per_cluster * per_group)
    ]
    group_labels = [
        f"g{index}" for index in range(n_clusters * groups_per_cluster) for _ in range(per_group)
    ]
    groups, n_groups = index_of(group_labels)
    clusters, n_cluster_codes = index_of(cluster_labels)

    x = rng.normal(size=n_obs)
    slope_by_cluster = 0.3 + rng.normal(scale=0.4, size=n_clusters)
    outcome = (
        np.repeat(rng.normal(scale=5.0, size=n_groups), per_group)
        + slope_by_cluster[clusters] * x
        + rng.normal(scale=0.2, size=n_obs)
    )
    regressors = x.reshape(-1, 1)

    clustered = fit_within_ols(
        outcome,
        regressors,
        ("x",),
        groups=groups,
        n_groups=n_groups,
        clusters=clusters,
        n_clusters=n_cluster_codes,
    )
    independent = fit_within_ols(
        outcome,
        regressors,
        ("x",),
        groups=groups,
        n_groups=n_groups,
        clusters=np.arange(n_obs, dtype=np.int64),
        n_clusters=n_obs,
    )
    assert clustered.std_errors[0] > 3.0 * independent.std_errors[0]
    assert clustered.degrees_of_freedom == n_clusters - 1


def test_the_within_transform_removes_the_group_mean() -> None:
    values = np.array([[1.0], [3.0], [10.0], [20.0]])
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    assert within_transform(values, groups, 2).ravel() == pytest.approx([-1.0, 1.0, -5.0, 5.0])


def test_a_rank_deficient_design_is_refused() -> None:
    """A collinear column is not identified, so returning a number for it would be a lie."""
    outcome, regressors, group_labels, cluster_labels = synthetic()
    collinear = np.column_stack([regressors[:, 0], regressors[:, 0] * 2.0])
    groups, n_groups = index_of(group_labels)
    clusters, n_clusters = index_of(cluster_labels)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_within_ols(
            outcome,
            collinear,
            ("a", "b"),
            groups=groups,
            n_groups=n_groups,
            clusters=clusters,
            n_clusters=n_clusters,
        )


def test_a_single_cluster_is_refused() -> None:
    outcome, regressors, group_labels, _ = synthetic()
    groups, n_groups = index_of(group_labels)
    with pytest.raises(ValueError, match="at least two clusters"):
        fit_within_ols(
            outcome,
            regressors,
            ("a", "b", "c"),
            groups=groups,
            n_groups=n_groups,
            clusters=np.zeros(outcome.size, dtype=np.int64),
            n_clusters=1,
        )


def test_a_design_with_too_few_observations_is_refused() -> None:
    outcome = np.arange(4.0)
    regressors = np.eye(4)[:, :3]
    groups = np.array([0, 0, 1, 1], dtype=np.int64)
    clusters = np.array([0, 0, 1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="cannot support"):
        fit_within_ols(
            outcome,
            regressors,
            ("a", "b", "c"),
            groups=groups,
            n_groups=2,
            clusters=clusters,
            n_clusters=2,
        )
