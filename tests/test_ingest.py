"""Translation, storage and retrieval of race data, without touching the network.

The tests that matter here are about trust in the files on disk: that a stored lap means what
its column name says, that the holdout season cannot leak into a training query, that a wet race
is recognised, and that a schema change cannot leave a stale file looking current.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from apex.config import AppSettings, DataSettings
from apex.data import (
    LAP_SCHEMA,
    SCHEMA_VERSION,
    conform_lap_frame,
    event_slug,
    ingest_race,
    load_laps,
    load_manifest,
    manifest_frame,
)
from apex.data.ingest import classify_dryness, record_path, summarise, write_race
from apex.data.schema import SessionRecord
from apex.data.sources import RawRace, to_lap_frame
from tests.factories import lap_frame, raw_lap, raw_race, settings_in, stint

# --------------------------------------------------------------------------- translation


def test_translation_preserves_every_lap() -> None:
    """Ingestion records what happened; deciding what to model happens later, elsewhere."""
    raw = raw_race([raw_lap(LapNumber=float(n), TrackStatus="4") for n in range(1, 11)])
    assert to_lap_frame(raw).height == 10


def test_durations_become_seconds() -> None:
    frame = to_lap_frame(raw_race([raw_lap(LapTime=dt.timedelta(seconds=91.234))]))
    assert frame["lap_time_s"].item() == pytest.approx(91.234)
    assert frame["sector3_s"].item() == pytest.approx(30.5)


def test_the_frame_matches_the_declared_schema() -> None:
    frame = to_lap_frame(raw_race())
    assert frame.columns == list(LAP_SCHEMA)
    assert dict(frame.schema) == LAP_SCHEMA


def test_pit_flags_are_derived_from_the_pit_times() -> None:
    raw = raw_race(
        [
            raw_lap(LapNumber=5.0, PitInTime=dt.timedelta(seconds=1000)),
            raw_lap(LapNumber=6.0, PitOutTime=dt.timedelta(seconds=1030)),
            raw_lap(LapNumber=7.0),
        ]
    )
    frame = to_lap_frame(raw).sort("lap")
    assert frame["is_pit_in"].to_list() == [True, False, False]
    assert frame["is_pit_out"].to_list() == [False, True, False]


def test_grid_positions_are_joined_by_driver() -> None:
    raw = raw_race(
        [raw_lap(Driver="VER"), raw_lap(Driver="HAM")], grid={"VER": 1, "HAM": 3, "LEC": 2}
    )
    frame = to_lap_frame(raw).sort("driver")
    assert frame.select("driver", "grid_position").rows() == [("HAM", 3), ("VER", 1)]


def test_a_driver_absent_from_the_classification_keeps_a_null_grid_slot() -> None:
    """Better an honest null than a fabricated grid slot for a car that never started."""
    frame = to_lap_frame(raw_race([raw_lap(Driver="XXX")], grid={"VER": 1}))
    assert frame["grid_position"].item() is None


def test_a_missing_fastf1_column_fails_loudly() -> None:
    raw = raw_race()
    crippled = RawRace(
        event=raw.event,
        laps=raw.laps.drop("TyreLife"),
        results=raw.results,
        rainfall_fraction=raw.rainfall_fraction,
    )
    with pytest.raises(ValueError, match="TyreLife"):
        to_lap_frame(crippled)


# --------------------------------------------------------------------------- dry classification


def test_a_race_with_no_rain_and_no_wet_tyres_is_dry(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    wet_laps, is_dry = classify_dryness(settings, lap_frame(stint(10)), rainfall_fraction=0.0)
    assert (wet_laps, is_dry) == (0, True)


def test_a_single_wet_tyre_makes_the_race_wet(tmp_path: Path) -> None:
    """Teams fit wets for conditions the trackside weather station under-reports."""
    settings = settings_in(tmp_path)
    laps = lap_frame([*stint(10), *stint(2, first_lap=30, compound="INTERMEDIATE")])
    wet_laps, is_dry = classify_dryness(settings, laps, rainfall_fraction=0.0)
    assert (wet_laps, is_dry) == (2, False)


def test_rainfall_above_the_threshold_makes_the_race_wet(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    laps = lap_frame(stint(10))
    assert classify_dryness(settings, laps, rainfall_fraction=0.5)[1] is False
    assert classify_dryness(settings, laps, rainfall_fraction=0.05)[1] is True


# --------------------------------------------------------------------------- storage


def stored(settings: AppSettings, raw: RawRace) -> SessionRecord:
    laps = to_lap_frame(raw)
    record = summarise(settings, raw, laps)
    write_race(settings, record, laps)
    return record


def test_a_written_race_round_trips(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    record = stored(settings, raw_race([raw_lap(LapNumber=float(n)) for n in range(1, 21)]))

    assert record.n_laps == 20
    assert load_manifest(settings) == (record,)
    assert load_laps(settings).height == 20


def test_the_sidecar_stamps_the_code_and_configuration_that_produced_the_file(
    tmp_path: Path,
) -> None:
    settings = settings_in(tmp_path)
    record = stored(settings, raw_race())
    on_disk = SessionRecord.model_validate_json(
        record_path(settings, record.key).read_text(encoding="utf-8")
    )
    assert on_disk.config_hash == settings.config_hash()
    assert on_disk.schema_version == SCHEMA_VERSION
    assert on_disk.fastf1_version
    assert on_disk.ingested_at.tzinfo is not None


def test_races_are_stored_one_file_each(tmp_path: Path) -> None:
    """One file per race is what lets an interrupted season pull resume where it stopped."""
    settings = settings_in(tmp_path)
    stored(settings, raw_race(round=1, event="Bahrain Grand Prix"))
    stored(settings, raw_race(round=6, event="Monaco Grand Prix"))
    keys = [record.key for record in load_manifest(settings)]
    assert keys == ["2023-01-bahrain-grand-prix", "2023-06-monaco-grand-prix"]


def test_the_manifest_is_ordered_by_season_then_round(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    stored(settings, raw_race(season=2024, round=2, event="Later Grand Prix"))
    stored(settings, raw_race(season=2023, round=9, event="Earlier Grand Prix"))
    frame = manifest_frame(settings)
    assert frame.select("season", "round").rows() == [(2023, 9), (2024, 2)]


def test_an_empty_store_still_has_the_right_shape(tmp_path: Path) -> None:
    """Downstream code should not need a special case for "nothing ingested yet"."""
    settings = settings_in(tmp_path)
    assert load_manifest(settings) == ()
    assert dict(load_laps(settings).schema) == LAP_SCHEMA
    assert load_laps(settings).height == 0


# --------------------------------------------------------------------------- retrieval


def race_over(settings: AppSettings, **overrides: Any) -> None:
    stored(settings, raw_race([raw_lap()], **overrides))


def test_loading_restricted_to_training_seasons_cannot_see_the_holdout(tmp_path: Path) -> None:
    """This filter is the mechanism behind the temporal split in ADR-0002."""
    settings = settings_in(tmp_path)
    for season in (2023, 2024, 2025):
        race_over(settings, season=season, event=f"{season} Grand Prix")

    train = load_laps(settings, seasons=settings.data.train_seasons)
    assert set(train["season"].to_list()) == {2023, 2024}


def test_wet_races_are_stored_but_excluded_from_dry_loads(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    stored(settings, raw_race([raw_lap()], round=1, event="Dry Grand Prix"))
    stored(
        settings,
        raw_race([raw_lap(Compound="WET")], round=2, event="Wet Grand Prix", rainfall_fraction=0.4),
    )

    assert len(load_manifest(settings)) == 2
    assert load_laps(settings, dry_only=True)["event"].unique().to_list() == ["Dry Grand Prix"]
    assert load_laps(settings, dry_only=False).height == 2


def test_loaded_laps_are_conformed_and_sorted(tmp_path: Path) -> None:
    settings = settings_in(tmp_path)
    stored(settings, raw_race([raw_lap(Driver="VER"), raw_lap(Driver="HAM")], round=2))
    stored(settings, raw_race([raw_lap(Driver="VER")], round=1, event="First Grand Prix"))
    laps = load_laps(settings)
    assert laps.select("round", "driver").rows() == [(1, "VER"), (2, "HAM"), (2, "VER")]
    assert dict(laps.schema) == LAP_SCHEMA


# --------------------------------------------------------------------------- re-ingestion


def test_an_ingested_race_is_not_downloaded_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_in(tmp_path)
    fetched: list[int] = []

    def fake_fetch(season: int, event: str | int, cache_dir: Path) -> RawRace:
        del event, cache_dir
        fetched.append(season)
        return raw_race()

    monkeypatch.setattr("apex.data.ingest.fetch_race", fake_fetch)
    monkeypatch.setattr("apex.data.ingest.scheduled_race", lambda *_, **__: raw_race().event)

    first = ingest_race(2023, "Test Grand Prix", settings)
    second = ingest_race(2023, "Test Grand Prix", settings)
    assert second.ingested_at == first.ingested_at
    assert fetched == [2023], "a race already on disk must not be loaded again"

    third = ingest_race(2023, "Test Grand Prix", settings, force=True)
    assert third.ingested_at > first.ingested_at
    assert len(fetched) == 2


def test_changing_the_dry_race_rule_invalidates_stored_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dry/wet verdict is decided at ingestion, so its rule is part of the file's identity."""
    settings = settings_in(tmp_path)
    monkeypatch.setattr("apex.data.ingest.fetch_race", lambda *_, **__: raw_race())
    monkeypatch.setattr("apex.data.ingest.scheduled_race", lambda *_, **__: raw_race().event)
    original = ingest_race(2023, "Test Grand Prix", settings)

    relaxed = settings_in(tmp_path, data=DataSettings(max_rainfall_fraction=0.5))
    refreshed = ingest_race(2023, "Test Grand Prix", relaxed)
    assert refreshed.ingested_at > original.ingested_at
    assert refreshed.data_fingerprint != original.data_fingerprint


def test_moving_the_train_holdout_split_does_not_invalidate_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which seasons are trained on is a query over stored races, not a property of one."""
    settings = settings_in(tmp_path)
    monkeypatch.setattr("apex.data.ingest.fetch_race", lambda *_, **__: raw_race())
    monkeypatch.setattr("apex.data.ingest.scheduled_race", lambda *_, **__: raw_race().event)
    original = ingest_race(2023, "Test Grand Prix", settings)

    resplit = settings_in(
        tmp_path, data=DataSettings(train_seasons=(2022, 2023), holdout_seasons=(2024, 2025))
    )
    assert ingest_race(2023, "Test Grand Prix", resplit).ingested_at == original.ingested_at


def test_an_unreadable_sidecar_is_treated_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_in(tmp_path)
    monkeypatch.setattr("apex.data.ingest.fetch_race", lambda *_, **__: raw_race())
    monkeypatch.setattr("apex.data.ingest.scheduled_race", lambda *_, **__: raw_race().event)
    original = ingest_race(2023, "Test Grand Prix", settings)
    record_path(settings, original.key).write_text('{"not": "a record"}', encoding="utf-8")

    assert ingest_race(2023, "Test Grand Prix", settings).ingested_at > original.ingested_at


def test_a_schema_change_invalidates_stored_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale file that still looks current is the failure mode this guards against."""
    settings = settings_in(tmp_path)
    monkeypatch.setattr("apex.data.ingest.fetch_race", lambda *_, **__: raw_race())
    monkeypatch.setattr("apex.data.ingest.scheduled_race", lambda *_, **__: raw_race().event)

    original = ingest_race(2023, "Test Grand Prix", settings)
    path = record_path(settings, original.key)
    path.write_text(
        original.model_copy(update={"schema_version": SCHEMA_VERSION - 1}).model_dump_json(),
        encoding="utf-8",
    )

    refreshed = ingest_race(2023, "Test Grand Prix", settings)
    assert refreshed.schema_version == SCHEMA_VERSION
    assert refreshed.ingested_at > original.ingested_at


# --------------------------------------------------------------------------- schema helpers


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ("Bahrain Grand Prix", "bahrain-grand-prix"),
        ("São Paulo Grand Prix", "s-o-paulo-grand-prix"),
        ("Miami  Grand   Prix", "miami-grand-prix"),
    ],
)
def test_event_slugs_are_filesystem_safe(event: str, expected: str) -> None:
    assert event_slug(event) == expected


def test_an_unusable_event_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="no usable characters"):
        event_slug("!!!")


def test_conforming_reorders_and_casts() -> None:
    frame = pl.DataFrame({name: [None] for name in reversed(list(LAP_SCHEMA))})
    conformed = conform_lap_frame(frame)
    assert conformed.columns == list(LAP_SCHEMA)
    assert dict(conformed.schema) == LAP_SCHEMA


def test_the_split_configuration_the_loader_relies_on_is_disjoint() -> None:
    data = DataSettings()
    assert not set(data.train_seasons) & set(data.holdout_seasons)
