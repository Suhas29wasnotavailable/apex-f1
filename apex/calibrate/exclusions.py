"""Deciding which laps may inform the pace model, and reporting what that cost.

A lap time is only evidence about tyre degradation if nothing else dominated it. Safety cars,
pit lane travel, deleted laps, spins and standing starts all produce lap times that are real but
uninformative, and a degradation slope fitted through them is confidently wrong.

Two commitments make the filtering honest rather than convenient:

This is the green-flag pace filter specifically. Pit loss and safety-car hazard are estimated
from the events these rules deliberately remove, so those models do their own, looser filtering
against the unfiltered race rather than reusing this one.

*   **Rules are declared in configuration, not chosen per circuit.** The same pipeline runs on
    every race. There is no hand-picking of incidents.
*   **Every rule reports what it removed.** :func:`clean_laps` returns a diagnostics table
    counting the laps each rule dropped and how many survived it, and that table is published
    alongside the fitted parameters. A filter that quietly discards 60% of a circuit's laps is
    then visible instead of being buried in a fit.

Counts are attributed in pipeline order: a lap removed as an in-lap is not counted again as a
safety-car lap. The order is fixed and meaningful -- the outlier rule runs late so that the
median it compares against is already a clean-lap median, and the stint-length rule runs last so
that it counts only laps that survived everything else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import polars as pl

from apex.config import AppSettings, get_settings

__all__ = ["CleaningReport", "clean_laps", "safety_car_laps"]

RACE_KEYS: Final = ("season", "round")
"""Columns identifying one race."""

CAR_KEYS: Final = ("season", "round", "driver")
"""Columns identifying one car's run through one race."""

STINT_KEYS: Final = ("season", "round", "driver", "stint")
"""Columns identifying one stint."""

_AFTER_SAFETY_CAR: Final = "_after_safety_car"


@dataclass(frozen=True, slots=True)
class CleaningReport:
    """The laps that survived, and the account of what did not.

    Attributes:
        laps: Surviving laps, in the input's schema.
        diagnostics: One row per rule -- ``rule``, ``removed``, ``remaining`` -- in the order the
            rules ran, preceded by an ``input`` row giving the starting count.
    """

    laps: pl.DataFrame
    diagnostics: pl.DataFrame

    @property
    def retained_fraction(self) -> float:
        """Share of input laps that survived. Zero for an empty input."""
        started = int(self.diagnostics["remaining"][0]) if self.diagnostics.height else 0
        return self.laps.height / started if started else 0.0


def _contains_any(column: str, codes: tuple[str, ...]) -> pl.Expr:
    """True where the string column contains any of ``codes`` as a substring.

    FastF1 concatenates every status that occurred during a lap, so a lap run partly under a
    virtual safety car reads ``"671"`` rather than ``"6"``. Substring matching is therefore the
    correct test, and matches FastF1's own ``Laps.pick_track_status`` semantics.
    """
    if not codes:
        return pl.lit(value=False)
    filled = pl.col(column).fill_null("")
    return pl.any_horizontal([filled.str.contains(code, literal=True) for code in codes])


def safety_car_laps(laps: pl.DataFrame, settings: AppSettings | None = None) -> pl.Expr:
    """Expression marking laps run wholly or partly under a safety car or VSC."""
    del laps
    settings = settings or get_settings()
    return _contains_any("track_status", settings.calibration.exclusions.safety_car_track_status)


def _flag_laps_after_safety_car(laps: pl.DataFrame, settings: AppSettings) -> pl.DataFrame:
    """Mark each lap whose immediately preceding lap, for the same car, was under a safety car.

    Computed on the unfiltered input: "the lap after" must mean the next lap the car actually
    ran, not the next lap that happened to survive an earlier rule.
    """
    return laps.sort(*CAR_KEYS, "lap").with_columns(
        safety_car_laps(laps, settings)
        .shift(1)
        .over(list(CAR_KEYS))
        .fill_null(value=False)
        .alias(_AFTER_SAFETY_CAR)
    )


def _outlier_threshold(ratio: float) -> pl.Expr:
    """Per-driver, per-race ceiling on lap time, relative to that car's median surviving lap."""
    return pl.col("lap_time_s").median().over(list(CAR_KEYS)) * ratio


def _rules(settings: AppSettings) -> list[tuple[str, Callable[[pl.DataFrame], pl.DataFrame]]]:
    """The ordered pipeline, with each enabled rule named for the diagnostics table."""
    exclusions = settings.calibration.exclusions
    calibration = settings.calibration
    rules: list[tuple[str, Callable[[pl.DataFrame], pl.DataFrame]]] = [
        # Not configurable: without a lap time there is no observation, and without a tyre age
        # the observation cannot be placed on a degradation curve. FastF1 leaves tyre age unknown
        # for a small number of laps where the timing feed never reported a stint change.
        ("missing_lap_time", lambda frame: frame.filter(pl.col("lap_time_s").is_not_null())),
        ("missing_tyre_age", lambda frame: frame.filter(pl.col("tyre_life").is_not_null())),
        (
            "wet_or_unknown_compound",
            lambda frame: frame.filter(pl.col("compound").is_in(calibration.dry_compounds)),
        ),
    ]
    if exclusions.require_accurate:
        rules.append(
            (
                "not_accurate",
                lambda frame: frame.filter(pl.col("is_accurate").fill_null(value=False)),
            )
        )
    if exclusions.drop_pit_in_out:
        rules.append(
            (
                "pit_in_or_out_lap",
                lambda frame: frame.filter(~pl.col("is_pit_in") & ~pl.col("is_pit_out")),
            )
        )
    if exclusions.drop_opening_laps:
        opening = exclusions.drop_opening_laps
        rules.append(("opening_lap", lambda frame: frame.filter(pl.col("lap") > opening)))
    if exclusions.excluded_track_status:
        codes = exclusions.excluded_track_status
        rules.append(
            ("track_status", lambda frame: frame.filter(~_contains_any("track_status", codes)))
        )
    if exclusions.drop_lap_after_safety_car:
        rules.append(
            ("lap_after_safety_car", lambda frame: frame.filter(~pl.col(_AFTER_SAFETY_CAR)))
        )
    if exclusions.drop_deleted:
        rules.append(
            (
                "deleted_by_stewards",
                lambda frame: frame.filter(~pl.col("deleted").fill_null(value=False)),
            )
        )
    if exclusions.drop_fastf1_generated:
        rules.append(
            (
                "fastf1_generated",
                lambda frame: frame.filter(~pl.col("fastf1_generated").fill_null(value=False)),
            )
        )
    ratio = exclusions.max_lap_time_ratio
    rules.append(
        (
            "slow_outlier",
            lambda frame: frame.filter(pl.col("lap_time_s") <= _outlier_threshold(ratio)),
        )
    )
    minimum = exclusions.min_clean_laps_per_stint
    rules.append(
        ("short_stint", lambda frame: frame.filter(pl.len().over(list(STINT_KEYS)) >= minimum))
    )
    return rules


def clean_laps(laps: pl.DataFrame, settings: AppSettings | None = None) -> CleaningReport:
    """Apply the configured exclusion rules, counting what each one removed.

    Args:
        laps: A canonical lap table, as written by :mod:`apex.data`.
        settings: Configuration; the process-wide settings by default.

    Returns:
        The surviving laps and a per-rule diagnostics table.
    """
    settings = settings or get_settings()
    frame = _flag_laps_after_safety_car(laps, settings)

    rows: list[dict[str, object]] = [{"rule": "input", "removed": 0, "remaining": frame.height}]
    for name, rule in _rules(settings):
        before = frame.height
        frame = rule(frame)
        rows.append({"rule": name, "removed": before - frame.height, "remaining": frame.height})

    return CleaningReport(
        laps=frame.drop(_AFTER_SAFETY_CAR).sort(*CAR_KEYS, "lap"),
        diagnostics=pl.DataFrame(
            rows, schema={"rule": pl.String(), "removed": pl.Int64(), "remaining": pl.Int64()}
        ),
    )
