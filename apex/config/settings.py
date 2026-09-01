"""Typed, validated, layered configuration.

No magic numbers in code: every threshold that shapes a result -- which seasons are trained on,
which laps are excluded from the pace model, how wide the dirty-air window is -- lives here, is
type-checked, and is hashed into the provenance stamp written alongside generated artefacts.

Precedence, highest first: explicit constructor arguments, ``APEX_``-prefixed environment
variables, then ``defaults.yaml``. Nested values use a double underscore, e.g.
``APEX_CALIBRATION__EXCLUSIONS__MAX_LAP_TIME_RATIO=1.05``.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from apex.seeding import SeedBundle

__all__ = ["PROJECT_ROOT", "AppSettings", "get_settings"]

PROJECT_ROOT: Final[Path] = Path(
    os.environ.get("APEX_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
"""Repository root. Overridable via ``APEX_PROJECT_ROOT`` so that containers and tests can
relocate generated artefacts without editing configuration files."""

DEFAULTS_PATH: Final[Path] = Path(__file__).with_name("defaults.yaml")

FIRST_TELEMETRY_SEASON: Final = 2018
"""FastF1's telemetry coverage begins in 2018; earlier seasons cannot be calibrated from."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PathSettings(_Frozen):
    """Filesystem locations for caches and generated artefacts.

    Relative values are resolved against :data:`PROJECT_ROOT`. Paths are excluded from the
    config hash, since they are machine-specific and must not change a result's identity.
    """

    fastf1_cache: Path = Path("data/fastf1_cache")
    processed: Path = Path("data/processed")
    params: Path = Path("params")
    figures: Path = Path("docs/figures")
    mlruns: Path = Path("mlruns")

    @field_validator("*", mode="after")
    @classmethod
    def _absolutise(cls, value: Path) -> Path:
        expanded = value.expanduser()
        return expanded if expanded.is_absolute() else PROJECT_ROOT / expanded

    def ensure_directories(self) -> None:
        """Create every configured directory. Safe to call repeatedly."""
        for path in self.model_dump().values():
            Path(path).mkdir(parents=True, exist_ok=True)


class DataSettings(_Frozen):
    """Which real sessions enter the pipeline, and the train/holdout split.

    The split is temporal rather than random: a twin is only useful if it generalises forward,
    so calibration never sees the holdout seasons.
    """

    train_seasons: tuple[int, ...] = (2022, 2023, 2024)
    holdout_seasons: tuple[int, ...] = (2025,)
    session: Literal["R"] = "R"
    pilot_events: tuple[str, ...] = ("Bahrain Grand Prix", "Monaco Grand Prix")
    dry_only: bool = True
    max_rainfall_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    wet_compounds: tuple[str, ...] = ("INTERMEDIATE", "WET")

    @property
    def all_seasons(self) -> tuple[int, ...]:
        return tuple(sorted({*self.train_seasons, *self.holdout_seasons}))

    def ingestion_fingerprint(self) -> str:
        """Hash of the settings that change what an ingested file contains.

        Deliberately narrow. Which seasons are trained on is a query over stored races, not a
        property of any stored race, so moving the split must not invalidate a single download.
        The dry-race rule is different: it is evaluated at ingestion and written into the
        sidecar, so changing it has to force those races to be reclassified.
        """
        payload = json.dumps(
            self.model_dump(
                mode="json", include={"session", "max_rainfall_fraction", "wet_compounds"}
            ),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @model_validator(mode="after")
    def _check_split(self) -> DataSettings:
        if not self.train_seasons:
            raise ValueError("train_seasons must not be empty")
        overlap = set(self.train_seasons) & set(self.holdout_seasons)
        if overlap:
            raise ValueError(f"train and holdout seasons overlap on {sorted(overlap)}")
        too_early = [s for s in self.all_seasons if s < FIRST_TELEMETRY_SEASON]
        if too_early:
            raise ValueError(
                f"seasons {too_early} predate FastF1 telemetry coverage "
                f"({FIRST_TELEMETRY_SEASON} onwards)"
            )
        return self


class ExclusionSettings(_Frozen):
    """Lap-level exclusion rules for the pace model.

    Each rule removes laps whose time reflects something other than the car's clean pace on its
    current tyre. The counts removed by each rule are reported in the calibration diagnostics
    rather than left implicit.
    """

    require_accurate: bool = True
    """Keep only laps FastF1 marks ``IsAccurate`` (timing synchronisation check)."""

    drop_pit_in_out: bool = True
    """Drop in- and out-laps, whose times include pit lane travel."""

    drop_opening_laps: int = Field(default=1, ge=0)
    """Drop the first N laps: a standing start is not representative green-flag pace."""

    excluded_track_status: tuple[str, ...] = ("2", "4", "5", "6", "7")
    """FastF1 track status codes to exclude: yellow, safety car, red, VSC deployed, VSC ending.
    Code ``1`` (all clear) is the only status retained."""

    drop_lap_after_safety_car: bool = True
    """The first green lap after a safety car period is a known source of timing artefacts."""

    safety_car_track_status: tuple[str, ...] = ("4", "6", "7")
    """Status codes that count as a safety car period for the rule above: safety car, virtual
    safety car deployed, virtual safety car ending. A subset of ``excluded_track_status``, named
    separately because the following lap is dropped for a different reason than the lap itself."""

    drop_deleted: bool = True
    """Drop laps deleted by the stewards, e.g. for track limits."""

    drop_fastf1_generated: bool = True
    """Drop laps FastF1 synthesised: partly interpolated, not measured."""

    max_lap_time_ratio: float = Field(default=1.10, gt=1.0)
    """Drop laps slower than this multiple of the driver's clean-lap median for that race, which
    removes spins, offs and damaged cars without hand-picking incidents."""

    min_clean_laps_per_stint: int = Field(default=4, ge=2)
    """A stint with fewer surviving laps cannot identify a degradation slope."""


class CalibrationSettings(_Frozen):
    """Model specification for the fitted parameters."""

    exclusions: ExclusionSettings = ExclusionSettings()
    dry_compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
    dirty_air_saturation_s: float = Field(default=2.0, gt=0.0)
    """Time constant of the saturating dirty-air penalty in the pace model. Traffic is
    controlled for rather than excluded, because the fitted penalty is itself a parameter the
    simulator needs."""

    degradation_degree: Literal[1, 2] = 2
    """Polynomial degree in tyre age; degree 2 allows a degradation cliff."""

    min_races_per_circuit: int = Field(default=2, ge=1)
    """Circuits below this are still fitted, but shrink hard toward the pooled compound mean."""

    min_laps_per_cell: int = Field(default=40, ge=10)
    """A circuit-and-compound cell below this is not fitted separately at all; it takes the
    pooled estimate. Below roughly this many laps the circuit fit is noise, and shrinkage on a
    noise estimate is still an estimate built on noise."""

    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    """Coverage of every published interval. One number, applied everywhere, so no interval in
    this project is quietly narrower than another."""


class AppSettings(BaseSettings):
    """Top-level configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="APEX_",
        env_nested_delimiter="__",
        yaml_file=DEFAULTS_PATH,
        yaml_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    seed: int = Field(default=20260901, ge=0, lt=2**64)
    paths: PathSettings = PathSettings()
    data: DataSettings = DataSettings()
    calibration: CalibrationSettings = CalibrationSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del dotenv_settings, file_secret_settings
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))

    def seeds(self) -> SeedBundle:
        """Root seed bundle for this configuration."""
        return SeedBundle(master_seed=self.seed)

    def config_hash(self) -> str:
        """Stable hash of everything that can change a result.

        Filesystem paths are excluded deliberately: the same experiment run in a container and
        on a laptop must produce the same stamp.
        """
        payload = json.dumps(self.model_dump(mode="json", exclude={"paths"}), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the process-wide settings, loaded once."""
    return AppSettings()
