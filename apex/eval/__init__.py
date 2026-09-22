"""Validation: how far the twin is from the races it has never seen.

The word "twin" is a claim about accuracy, so it stays unused until this package has produced
published error metrics against held-out races. Everything here reads
:class:`~apex.calibrate.params.ParameterSet` and the ingested holdout laps, and nothing here can
change a fitted parameter.
"""

from __future__ import annotations

from apex.eval.metrics import RaceComparison, compare
from apex.eval.observed import ObservedRace, ReplayStops, observed_race, observed_schedule
from apex.eval.validate import FreeSummary, ValidationReport, validate_holdout, validate_race

__all__ = [
    "FreeSummary",
    "ObservedRace",
    "RaceComparison",
    "ReplayStops",
    "ValidationReport",
    "compare",
    "observed_race",
    "observed_schedule",
    "validate_holdout",
    "validate_race",
]
