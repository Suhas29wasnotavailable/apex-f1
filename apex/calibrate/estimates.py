"""A fitted number and everything needed to judge it.

ADR-0002 commits to publishing no parameter without an uncertainty estimate. That commitment is
enforced by making :class:`Estimate` the only way a number leaves the calibration layer: it
carries the value, its standard error, an interval at the project's single configured coverage,
and the number of observations behind it.

An estimate whose uncertainty genuinely cannot be measured -- one observation, or one cluster --
says so with ``None`` rather than with an infinite interval. Infinity is not representable in
JSON, so a file containing one would not survive the round trip to the simulator, and "unknown"
is in any case the honest description of a spread that was never estimable. Downstream code has
to handle that case, which is the point: it cannot mistake an unmeasured spread for a narrow one.
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
    std_error: float | None = Field(default=None, ge=0.0)
    """``None`` when the spread could not be estimated at all."""

    interval: tuple[float, float] | None = None
    """``None`` when there were too few independent observations to support one."""

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
        """Build an estimate, deriving the interval from the standard error.

        With fewer than two independent observations there is no interval to derive, and the
        estimate is returned saying so rather than claiming one of infinite width.
        """
        dof = float(n_obs - 1) if degrees_of_freedom is None else degrees_of_freedom
        multiplier = critical_value(dof, level)
        interval = (
            None
            if math.isinf(multiplier)
            else (value - multiplier * std_error, value + multiplier * std_error)
        )
        return cls(
            value=value,
            std_error=std_error,
            interval=interval,
            level=level,
            n_obs=n_obs,
            shrinkage=shrinkage,
        )

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval separates the effect from nothing at all.

        False when there is no interval: an unmeasured spread is not evidence of an effect.
        """
        if self.interval is None:
            return False
        low, high = self.interval
        return low > 0.0 or high < 0.0

    def __str__(self) -> str:
        spread = "±   ?" if self.std_error is None else f"± {self.std_error:.4f}"
        bounds = (
            "[unmeasurable]"
            if self.interval is None
            else f"[{self.interval[0]:+.4f}, {self.interval[1]:+.4f}]"
        )
        return f"{self.value:+.4f} {spread} {bounds} n={self.n_obs}"
