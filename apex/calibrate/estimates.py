"""A fitted number and everything needed to judge it.

ADR-0002 commits to publishing no parameter without an uncertainty estimate. That commitment is
enforced by making :class:`Estimate` the only way a number leaves the calibration layer: it
carries the value, its standard error, an interval at the project's single configured coverage,
and the number of observations behind it. A point estimate with no interval cannot be
constructed, so it cannot be published by accident.
"""

from __future__ import annotations

import math
from typing import Final

from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

__all__ = ["Estimate", "critical_value"]

_MIN_DEGREES_OF_FREEDOM: Final = 1


def critical_value(degrees_of_freedom: float, level: float) -> float:
    """Two-sided t critical value, falling back to the normal limit for large samples.

    The t distribution rather than the normal because circuit-level fits routinely rest on a
    handful of races, where the difference is the difference between an honest interval and a
    flattering one.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence level must be in (0, 1), got {level}")
    if degrees_of_freedom < _MIN_DEGREES_OF_FREEDOM:
        return math.inf
    return float(stats.t.ppf(0.5 + level / 2.0, degrees_of_freedom))


class Estimate(BaseModel):
    """One fitted quantity, with its uncertainty and provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float
    std_error: float = Field(ge=0.0)
    interval: tuple[float, float]
    level: float = Field(gt=0.5, lt=1.0)
    n_obs: int = Field(ge=0)
    """Observations the estimate rests on. For a shrunk circuit estimate, the laps in that
    circuit's own cell -- not the pooled total, which would overstate what is known locally."""

    shrinkage: float | None = Field(default=None, ge=0.0, le=1.0)
    """Weight this estimate's own data carried after partial pooling: 1.0 is unshrunk, 0.0 is the
    pooled mean. ``None`` for an estimate that was never pooled."""

    @classmethod
    def from_standard_error(
        cls,
        value: float,
        std_error: float,
        *,
        n_obs: int,
        level: float,
        degrees_of_freedom: float | None = None,
        shrinkage: float | None = None,
    ) -> Estimate:
        """Build an estimate, deriving the interval from the standard error."""
        dof = float(n_obs - 1) if degrees_of_freedom is None else degrees_of_freedom
        half_width = critical_value(dof, level) * std_error
        return cls(
            value=value,
            std_error=std_error,
            interval=(value - half_width, value + half_width),
            level=level,
            n_obs=n_obs,
            shrinkage=shrinkage,
        )

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval separates the effect from nothing at all."""
        low, high = self.interval
        return low > 0.0 or high < 0.0

    def __str__(self) -> str:
        low, high = self.interval
        return f"{self.value:+.4f} ± {self.std_error:.4f} [{low:+.4f}, {high:+.4f}] n={self.n_obs}"
