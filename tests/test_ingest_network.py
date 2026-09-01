"""The one test that actually talks to FastF1.

Everything else in the suite runs against constructed frames, which proves the logic but cannot
prove that the assumptions about FastF1's data still hold. This test does exactly that, on one
race, and is excluded from CI by the ``network`` marker. Run it with ``pytest -m network`` after
changing anything in :mod:`apex.data.sources`.

The assertions are deliberately about facts of the 2023 Bahrain Grand Prix that a data-source
change would break -- twenty starters, fifty-seven laps, a dry race, lap times near 96 seconds --
rather than about exact values a FastF1 patch release could legitimately move.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from apex.calibrate.exclusions import clean_laps
from apex.config import PROJECT_ROOT, AppSettings, PathSettings
from apex.data import LAP_SCHEMA, SessionRecord, ingest_race, load_laps

pytestmark = [
    pytest.mark.network,
    pytest.mark.slow,
    # FastF1 warns about known timing-data quirks while parsing. They are the data source's
    # business, not this project's, and must not fail the test.
    pytest.mark.filterwarnings("default"),
]

SEASON = 2023
EVENT = "Bahrain Grand Prix"


@pytest.fixture
def settings(tmp_path: Path) -> AppSettings:
    """Write processed files to a temporary directory, but share the repository's FastF1 cache.

    Re-downloading a session that is already cached would be slow and would put avoidable load on
    the timing API for no additional coverage.
    """
    return AppSettings(
        paths=PathSettings(
            fastf1_cache=PROJECT_ROOT / "data" / "fastf1_cache",
            processed=tmp_path / "processed",
        )
    )


@pytest.fixture
def ingested(settings: AppSettings) -> tuple[SessionRecord, pl.DataFrame]:
    record = ingest_race(SEASON, EVENT, settings)
    return record, load_laps(settings, seasons=[SEASON])


def test_a_real_race_ingests_into_the_declared_schema(
    ingested: tuple[SessionRecord, pl.DataFrame],
) -> None:
    record, laps = ingested

    assert record.key == "2023-01-bahrain-grand-prix"
    assert record.event.total_laps == 57
    assert record.n_drivers == 20
    assert record.is_dry, "the 2023 Bahrain Grand Prix was dry"
    assert record.wet_compound_laps == 0

    assert dict(laps.schema) == LAP_SCHEMA
    assert laps.height == record.n_laps
    assert laps["lap"].max() == 57
    assert set(laps["compound"].unique()) <= {"SOFT", "MEDIUM", "HARD"}
    assert laps["grid_position"].null_count() == 0


def test_real_lap_times_are_plausible_seconds(
    ingested: tuple[SessionRecord, pl.DataFrame], settings: AppSettings
) -> None:
    """A units error here would be invisible in a fit and catastrophic in the simulator."""
    clean = clean_laps(ingested[1], settings).laps
    median = float(np.median(clean["lap_time_s"].to_numpy()))
    assert 90.0 < median < 105.0, f"Bahrain green-flag laps are around 96s, got {median}"


def test_the_exclusion_rules_keep_most_of_a_clean_race(
    ingested: tuple[SessionRecord, pl.DataFrame], settings: AppSettings
) -> None:
    """A dry race should lose laps to pit stops and neutralisations, but not most of them."""
    report = clean_laps(ingested[1], settings)

    assert 0.6 < report.retained_fraction < 1.0
    assert report.laps["stint"].n_unique() > 1
    assert report.laps["driver"].n_unique() >= 15


def test_the_track_status_rule_finds_the_real_neutralisations(
    ingested: tuple[SessionRecord, pl.DataFrame], settings: AppSettings
) -> None:
    """FastF1 concatenates statuses within a lap, so the rule must match codes as substrings."""
    laps = ingested[1]
    assert laps.filter(pl.col("track_status") != "1").height > 0

    report = clean_laps(laps, settings)
    assert report.laps["track_status"].unique().to_list() == ["1"]
