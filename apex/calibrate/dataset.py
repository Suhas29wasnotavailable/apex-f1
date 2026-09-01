"""Assembling the laps a fit is allowed to see.

The order of operations here is load-bearing and is the reason this is a function rather than
three calls at each call site. Features are derived on the unfiltered race, because the gap to
the car ahead depends on laps the exclusion rules will remove; the rules run afterwards on the
featured frame. Doing it the other way round would measure each car's gap to the nearest car
whose lap happened to survive, which is not a gap anybody drove.

The season filter is the mechanism behind ADR-0002's temporal split. Calibration calls
:func:`training_laps`, which cannot reach the holdout seasons; validation in P2 will call
:func:`holdout_laps`, which cannot reach the training ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from apex.calibrate.exclusions import clean_laps
from apex.calibrate.features import add_features
from apex.config import AppSettings, get_settings
from apex.data import load_laps

__all__ = ["CalibrationData", "holdout_laps", "prepare_laps", "training_laps"]


@dataclass(frozen=True, slots=True)
class CalibrationData:
    """Model-ready laps, with the account of everything that was excluded to get them.

    The diagnostics travel with the data rather than being computed on request, so that a fit
    cannot be reported without the filtering that produced its inputs.
    """

    laps: pl.DataFrame
    diagnostics: pl.DataFrame
    seasons: tuple[int, ...]

    @property
    def n_races(self) -> int:
        return self.laps.select("race_id").n_unique()

    @property
    def n_circuits(self) -> int:
        return self.laps.select("circuit").n_unique()

    def circuit(self, name: str) -> pl.DataFrame:
        return self.laps.filter(pl.col("circuit") == name)


def prepare_laps(
    seasons: Sequence[int], settings: AppSettings | None = None, *, dry_only: bool = True
) -> CalibrationData:
    """Load, feature and clean the laps for the given seasons."""
    settings = settings or get_settings()
    raw = load_laps(settings, seasons=seasons, dry_only=dry_only)
    report = clean_laps(add_features(raw, settings), settings)
    return CalibrationData(laps=report.laps, diagnostics=report.diagnostics, seasons=tuple(seasons))


def training_laps(settings: AppSettings | None = None) -> CalibrationData:
    """Every lap calibration is allowed to see. Never the holdout seasons."""
    settings = settings or get_settings()
    return prepare_laps(settings.data.train_seasons, settings)


def holdout_laps(settings: AppSettings | None = None) -> CalibrationData:
    """The held-out seasons, for validation only. No fitted parameter may be derived from these."""
    settings = settings or get_settings()
    return prepare_laps(settings.data.holdout_seasons, settings)
