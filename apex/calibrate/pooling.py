"""Partial pooling of circuit-level estimates toward their common mean.

ADR-0002's problem: circuits appear between two and eight times in the calibration window.
Fitting each independently gives Las Vegas an interval so wide it is useless; pooling them all
erases the difference between Monaco and Monza, which is precisely what race strategy turns on.

The compromise is empirical Bayes. Treat each circuit's true value as drawn from a distribution
around a common mean, estimate that distribution's spread from the disagreement between circuits
that is larger than their own standard errors explain, and pull each circuit toward the mean in
proportion to how little its own data says. A circuit with eight races barely moves; a circuit
with two races moves most of the way. Nothing is hand-tuned: the shrinkage weight falls out of
the estimates' own standard errors, and it is published alongside every value so a reader can
see how much of a circuit's number came from that circuit.

The between-circuit variance uses the DerSimonian-Laird moment estimator, and each shrunk
interval includes the uncertainty in the pooled mean itself rather than conditioning on it as
though it were known.

One deliberate departure from textbook empirical Bayes. The method assumes each group's standard
error is known, and here it is not: a circuit with two races has a cluster-robust standard error
estimated from two clusters, which can come out far too small by luck. Weighing evidence by that
number lets a circuit with almost no data refuse to shrink -- the opposite of what pooling is
for. So each group's evidence is measured by the *width of its interval* rather than by its
standard error alone. The interval already accounts for the degrees of freedom behind it, so a
two-race circuit is correctly treated as knowing roughly a sixth of what its standard error
claims, and a circuit with no usable interval at all contributes nothing and takes the mean.
Because the inputs are then normal-equivalent, the outputs use the normal critical value rather
than applying a second small-sample correction on top of the first.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from apex.calibrate.estimates import Estimate, critical_value

__all__ = ["PartialPooling", "effective_standard_error", "partial_pool"]

_MIN_STANDARD_ERROR = 1e-9
"""Floor on a reported standard error, so that a degenerate fit cannot claim infinite precision
and dominate the pooled mean."""


def effective_standard_error(estimate: Estimate) -> float:
    """The standard error a normal interval of the same width would have.

    This is how much a group is treated as knowing. For a fit resting on many clusters it is
    the standard error itself; for one resting on two, it is several times larger, because that
    is what the interval actually says. Infinite when the estimate has no usable interval.
    """
    low, high = estimate.interval
    if not math.isfinite(low) or not math.isfinite(high):
        return math.inf
    return max((high - low) / (2.0 * critical_value(math.inf, estimate.level)), _MIN_STANDARD_ERROR)


@dataclass(frozen=True, slots=True)
class PartialPooling:
    """The result of pooling a family of estimates.

    Attributes:
        mean: The pooled mean, with its own uncertainty.
        between_std: Estimated standard deviation of true values across groups. Zero means the
            groups disagree no more than their own noise explains, and every group collapses onto
            the mean.
        estimates: Shrunk estimate per group, each carrying its shrinkage weight.
    """

    mean: Estimate
    between_std: float
    estimates: dict[str, Estimate]


def partial_pool(observations: Mapping[str, Estimate], *, level: float) -> PartialPooling:
    """Shrink per-group estimates toward their precision-weighted common mean.

    Args:
        observations: Independent per-group estimates, keyed by group.
        level: Coverage for the returned intervals.

    Raises:
        ValueError: if there is nothing to pool.
    """
    if not observations:
        raise ValueError("nothing to pool")

    keys = list(observations)
    values = np.array([observations[key].value for key in keys], dtype=np.float64)
    variances = np.array(
        [effective_standard_error(observations[key]) ** 2 for key in keys], dtype=np.float64
    )
    counts = [observations[key].n_obs for key in keys]
    if not np.isfinite(variances).any():
        raise ValueError("no group has a usable interval, so there is nothing to pool")

    if len(keys) == 1:
        # One group cannot inform a between-group spread, so there is nothing to shrink toward
        # and the estimate is returned as it came in.
        only = observations[keys[0]]
        return PartialPooling(
            mean=only,
            between_std=0.0,
            estimates={keys[0]: only.model_copy(update={"shrinkage": 1.0})},
        )

    between_variance = _between_variance(values, variances)

    weights = 1.0 / (variances + between_variance)
    pooled_mean = float(values @ weights / weights.sum())
    pooled_variance = float(1.0 / weights.sum())

    shrinkage = between_variance / (between_variance + variances)
    shrunk_values = shrinkage * values + (1.0 - shrinkage) * pooled_mean
    # Posterior variance given the mean, plus the mean's own variance, which a reader is
    # otherwise invited to forget was estimated too. A group with no usable interval has zero
    # shrinkage and contributes none of its own variance, so it is simply the pooled mean.
    own_variance = np.zeros_like(variances)
    informative = np.isfinite(variances)
    own_variance[informative] = shrinkage[informative] * variances[informative]
    shrunk_variances = own_variance + (1.0 - shrinkage) ** 2 * pooled_variance

    return PartialPooling(
        mean=Estimate.from_standard_error(
            pooled_mean,
            float(np.sqrt(pooled_variance)),
            n_obs=sum(counts),
            level=level,
            degrees_of_freedom=math.inf,
        ),
        between_std=float(np.sqrt(between_variance)),
        estimates={
            key: Estimate.from_standard_error(
                float(shrunk_values[position]),
                float(np.sqrt(shrunk_variances[position])),
                n_obs=counts[position],
                level=level,
                degrees_of_freedom=math.inf,
                shrinkage=float(shrinkage[position]),
            )
            for position, key in enumerate(keys)
        },
    )


def _between_variance(values: np.ndarray, variances: np.ndarray) -> float:
    """DerSimonian-Laird moment estimator of the between-group variance.

    Returns zero when the observed disagreement between groups is no larger than their own
    standard errors already account for, which is the honest reading of that evidence: there is
    no detectable circuit effect, so every circuit takes the pooled value.
    """
    precisions = 1.0 / variances
    if precisions.sum() <= 0.0:
        return 0.0
    fixed_effect_mean = float(values @ precisions / precisions.sum())
    dispersion = float(precisions @ (values - fixed_effect_mean) ** 2)
    scale = float(precisions.sum() - (precisions**2).sum() / precisions.sum())
    if scale <= 0.0:
        return 0.0
    return max(0.0, (dispersion - (values.size - 1)) / scale)
