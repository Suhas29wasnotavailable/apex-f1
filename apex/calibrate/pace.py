"""The pace model: what makes a lap slower than the same car's own average lap.

    lap_time = (this car, this race) + compound offset + degradation(tyre age)
               + progress(fuel and track evolution) + traffic(gap to the car ahead) + noise

Everything constant about a car in a race -- the machinery, the driver, the circuit, the day's
track temperature -- is absorbed by a fixed effect rather than modelled, because none of it is a
parameter the simulator needs and all of it would otherwise contaminate the parameters that are.
What survives is the handful of effects a strategy decision actually trades against each other.

Identification rests on stints. Within one stint, tyre age and fuel burn both advance one lap at
a time and cannot be told apart. Across a pit stop they separate: tyre age resets, fuel does not.
A driver's second stint therefore carries the information that fixes the progress effect, and it
is the reason the degradation slope here is not simply the whole of a stint's slowdown.

Compound level and compound slope are estimated separately. "The hard tyre is slower" and "the
hard tyre degrades more slowly" are different claims, they point in opposite directions for
strategy, and a model without the offset term silently mixes them into one number.

Traffic is controlled for rather than excluded, per ADR-0002. Dropping laps run in traffic would
bias degradation toward cars in clean air -- disproportionately the fast ones on fresh tyres --
and would throw away the dirty-air penalty the simulator has to have.

Circuits are fitted separately and then pooled. Monaco and Monza genuinely differ in every one of
these coefficients, but half the circuits in the window appear two or three times, so an
independent fit would give them unusable intervals. Each circuit is shrunk toward the common mean
in proportion to how little its own data says, and the weight it kept is published with it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from apex.calibrate.estimates import Estimate
from apex.calibrate.pooling import partial_pool
from apex.calibrate.regression import RegressionResult, fit_within_ols, index_of
from apex.config import AppSettings, get_settings

__all__ = [
    "CircuitPace",
    "CompoundDegradation",
    "PaceModel",
    "compound_offset_term",
    "degradation_term",
    "fit_circuit_pace",
    "fit_pace",
]

FUEL = "race_progress"
DIRTY_AIR = "dirty_air"


def degradation_term(compound: str, degree: int) -> str:
    """Name of the regressor for one compound's tyre-age term of a given degree."""
    return f"age{degree}_{compound}"


def compound_offset_term(compound: str) -> str:
    """Name of the regressor for one compound's pace offset against the reference compound."""
    return f"offset_{compound}"


class CompoundDegradation(BaseModel):
    """How much lap time one compound loses as its tyres age, at one circuit or pooled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    compound: str
    circuit: str | None = None
    """``None`` for the estimate pooled across every circuit in the training seasons."""

    coefficients: tuple[Estimate, ...] = Field(min_length=1)
    """Polynomial coefficients in tyre age, lowest degree first: index 0 multiplies age, index 1
    multiplies age squared. There is no constant term -- the compound's own pace level is
    :attr:`offset_s`, and a fresh tyre of that compound defines zero *additional* loss."""

    offset_s: Estimate | None = None
    """Lap time relative to the reference compound at equal tyre age. ``None`` for the reference
    compound itself, which is the zero this is measured against."""

    def seconds_lost(self, age: float) -> float:
        """Lap time lost to tyre age alone, relative to a fresh tyre of the same compound."""
        return float(
            sum(
                estimate.value * age ** (degree + 1)
                for degree, estimate in enumerate(self.coefficients)
            )
        )

    def lap_time_delta(self, age: float) -> float:
        """Lap time relative to a fresh reference-compound tyre: the offset plus the degradation."""
        offset = 0.0 if self.offset_s is None else self.offset_s.value
        return offset + self.seconds_lost(age)


class PaceModel(BaseModel):
    """A fitted pace model and the evidence behind it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    circuit: str | None = None
    """``None`` for the model pooled across every circuit."""

    degradation: tuple[CompoundDegradation, ...]

    reference_compound: str
    """The compound every offset is measured against, and the one whose offset is zero."""

    fuel_effect_s: Estimate
    """Lap time difference between the start and the end of a race attributable to race progress.
    Expected to be negative: the car burns fuel and gets lighter. It also absorbs track evolution,
    which moves the same way over a race and is not separable from fuel burn by this design, so it
    is a combined progress effect rather than a pure fuel coefficient."""

    dirty_air_s: Estimate
    """Lap time lost running immediately behind another car, decaying with the gap."""

    dirty_air_saturation_s: float
    """The time constant the penalty decays over, echoed from configuration so that the fitted
    coefficient can never be read without the shape it belongs to."""

    residual_std_s: float
    r_squared_within: float
    n_laps: int
    n_races: int
    n_cars: int

    def dirty_air_penalty(self, gap_s: float) -> float:
        """Seconds lost at a given gap to the car ahead."""
        decay = float(np.exp(-max(gap_s, 0.0) / self.dirty_air_saturation_s))
        return self.dirty_air_s.value * decay

    def compound(self, name: str) -> CompoundDegradation:
        for entry in self.degradation:
            if entry.compound == name:
                return entry
        raise KeyError(f"no degradation fitted for compound {name!r}")


@dataclass(frozen=True, slots=True)
class CircuitPace:
    """Per-circuit pace models after partial pooling, and the account of how they were made.

    Attributes:
        pooled: The model fitted to every training lap at once.
        circuits: One model per circuit, with every coefficient shrunk toward the pooled mean.
        between_std: Estimated standard deviation of each coefficient across circuits. A value of
            zero means the circuits disagreed no more than their own noise explains, so that
            coefficient is effectively constant across the calendar.
        fitted: Circuits with enough races and laps to estimate anything of their own.
        pooled_only: Circuits that took the pooled values wholesale. Their estimates are reported
            with a shrinkage weight of zero, which is the honest label for a number that contains
            none of that circuit's own data.
    """

    pooled: PaceModel
    circuits: dict[str, PaceModel]
    between_std: dict[str, float]
    fitted: tuple[str, ...]
    pooled_only: tuple[str, ...]


# --------------------------------------------------------------------------------- fitting


def _fittable_compounds(laps: pl.DataFrame, settings: AppSettings) -> tuple[str, ...]:
    """Compounds with enough laps here to support their own degradation curve."""
    minimum = settings.calibration.min_laps_per_cell
    counts = dict(laps.group_by("compound").len().iter_rows())
    return tuple(
        name for name in settings.calibration.dry_compounds if int(counts.get(name, 0)) >= minimum
    )


def _design(
    laps: pl.DataFrame, compounds: Sequence[str], degree: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the regressor matrix.

    Columns are, in order: an offset for every compound but the first, which is the reference; a
    tyre-age polynomial per compound; race progress; traffic. There is no intercept -- the
    car-race fixed effects supply one, and a second would be collinear with them.
    """
    age = laps["tyre_age"].to_numpy()
    indicators = {
        compound: (laps["compound"] == compound).to_numpy().astype(np.float64)
        for compound in compounds
    }
    columns: list[np.ndarray] = []
    names: list[str] = []
    for compound in compounds[1:]:
        columns.append(indicators[compound])
        names.append(compound_offset_term(compound))
    for compound in compounds:
        for power in range(1, degree + 1):
            columns.append(indicators[compound] * age**power)
            names.append(degradation_term(compound, power))
    columns.append(laps[FUEL].to_numpy())
    names.append(FUEL)
    columns.append(laps[DIRTY_AIR].to_numpy())
    names.append(DIRTY_AIR)
    return np.column_stack(columns), tuple(names)


def _fit(
    laps: pl.DataFrame, compounds: Sequence[str], settings: AppSettings
) -> tuple[RegressionResult, pl.DataFrame]:
    """Run the regression on the laps of the given compounds."""
    usable = laps.filter(pl.col("compound").is_in(compounds))
    regressors, names = _design(usable, compounds, settings.calibration.degradation_degree)
    groups, n_groups = index_of(usable["car_id"].to_list())
    clusters, n_clusters = index_of(usable["race_id"].to_list())
    result = fit_within_ols(
        usable["lap_time_s"].to_numpy(),
        regressors,
        names,
        groups=groups,
        n_groups=n_groups,
        clusters=clusters,
        n_clusters=n_clusters,
    )
    return result, usable


def _coefficients(
    result: RegressionResult, laps: pl.DataFrame, compounds: Sequence[str], level: float
) -> dict[str, Estimate]:
    """Every coefficient as an :class:`Estimate`, each carrying the laps behind it.

    A compound's coefficients report that compound's lap count rather than the fit's total, so a
    number resting on three hundred laps is never displayed as though it rested on thirty
    thousand.
    """
    counts = {name: int(value) for name, value in laps.group_by("compound").len().iter_rows()}
    estimates: dict[str, Estimate] = {}
    for name in result.names:
        compound = next((entry for entry in compounds if name.endswith(f"_{entry}")), None)
        n_obs = counts.get(compound, result.n_obs) if compound else result.n_obs
        estimates[name] = result.estimate(name, level, n_obs=n_obs)
    return estimates


def _model(
    coefficients: Mapping[str, Estimate],
    compounds: Sequence[str],
    settings: AppSettings,
    *,
    circuit: str | None,
    residual_std_s: float,
    r_squared_within: float,
    n_laps: int,
    n_races: int,
    n_cars: int,
) -> PaceModel:
    """Assemble a :class:`PaceModel` from named coefficients, whatever produced them."""
    degree = settings.calibration.degradation_degree
    return PaceModel(
        circuit=circuit,
        degradation=tuple(
            CompoundDegradation(
                compound=compound,
                circuit=circuit,
                coefficients=tuple(
                    coefficients[degradation_term(compound, power)]
                    for power in range(1, degree + 1)
                ),
                offset_s=(None if position == 0 else coefficients[compound_offset_term(compound)]),
            )
            for position, compound in enumerate(compounds)
        ),
        reference_compound=compounds[0],
        fuel_effect_s=coefficients[FUEL],
        dirty_air_s=coefficients[DIRTY_AIR],
        dirty_air_saturation_s=settings.calibration.dirty_air_saturation_s,
        residual_std_s=residual_std_s,
        r_squared_within=r_squared_within,
        n_laps=n_laps,
        n_races=n_races,
        n_cars=n_cars,
    )


def fit_pace(laps: pl.DataFrame, settings: AppSettings | None = None) -> PaceModel:
    """Fit one pace model to every training lap.

    Args:
        laps: Cleaned laps carrying the features from :func:`~apex.calibrate.features.add_features`
            -- in practice, ``training_laps(settings).laps``.
        settings: Configuration; the process-wide settings by default.

    Raises:
        ValueError: if the laps cannot identify the model, rather than returning a number the
            data does not support.
    """
    settings = settings or get_settings()
    compounds = _fittable_compounds(laps, settings)
    if not compounds:
        raise ValueError(
            f"no compound has the {settings.calibration.min_laps_per_cell} laps required to fit "
            f"a degradation curve"
        )
    result, usable = _fit(laps, compounds, settings)
    return _model(
        _coefficients(result, usable, compounds, settings.calibration.confidence_level),
        compounds,
        settings,
        circuit=None,
        residual_std_s=result.residual_std,
        r_squared_within=result.r_squared_within,
        n_laps=result.n_obs,
        n_races=result.n_clusters,
        n_cars=result.n_absorbed,
    )


# --------------------------------------------------------------------------------- pooling


@dataclass(frozen=True, slots=True)
class _CircuitFit:
    """One circuit's own fit, before any pooling."""

    circuit: str
    compounds: tuple[str, ...]
    coefficients: dict[str, Estimate]
    laps: int
    races: int
    cars: int
    residual_std_s: float
    r_squared_within: float


def _fit_circuit(laps: pl.DataFrame, circuit: str, settings: AppSettings) -> _CircuitFit | None:
    """Fit one circuit, or return None if its data cannot identify anything of its own."""
    calibration = settings.calibration
    at_circuit = laps.filter(pl.col("circuit") == circuit)
    if at_circuit.select("race_id").n_unique() < calibration.min_races_per_circuit:
        return None
    compounds = _fittable_compounds(at_circuit, settings)
    if not compounds:
        return None
    try:
        result, usable = _fit(at_circuit, compounds, settings)
    except (ValueError, np.linalg.LinAlgError):
        # A circuit whose laps are collinear -- every stint the same length, say -- has nothing
        # of its own to contribute, and pretending otherwise is what shrinkage exists to avoid.
        return None
    return _CircuitFit(
        circuit=circuit,
        compounds=compounds,
        coefficients=_coefficients(result, usable, compounds, calibration.confidence_level),
        laps=result.n_obs,
        races=result.n_clusters,
        cars=result.n_absorbed,
        residual_std_s=result.residual_std,
        r_squared_within=result.r_squared_within,
    )


def fit_circuit_pace(laps: pl.DataFrame, settings: AppSettings | None = None) -> CircuitPace:
    """Fit the pooled model and one partially-pooled model per circuit.

    Args:
        laps: Cleaned, featured training laps.
        settings: Configuration; the process-wide settings by default.
    """
    settings = settings or get_settings()
    level = settings.calibration.confidence_level
    pooled = fit_pace(laps, settings)
    compounds = tuple(entry.compound for entry in pooled.degradation)

    circuits = sorted(laps["circuit"].unique().to_list())
    fits = {
        circuit: fit
        for circuit in circuits
        if (fit := _fit_circuit(laps, circuit, settings)) is not None
    }

    global_coefficients = _pooled_coefficients(pooled, compounds, settings)
    shrunk: dict[str, dict[str, Estimate]] = {}
    between_std: dict[str, float] = {}
    for name in global_coefficients:
        observed = {
            circuit: fit.coefficients[name]
            for circuit, fit in fits.items()
            if name in fit.coefficients
        }
        if not observed:
            # No circuit could estimate this coefficient on its own, so every circuit takes the
            # pooled value and there is no between-circuit spread to report.
            between_std[name] = 0.0
            continue
        pooling = partial_pool(observed, level=level, toward=global_coefficients[name])
        between_std[name] = pooling.between_std
        for circuit, estimate in pooling.estimates.items():
            shrunk.setdefault(circuit, {})[name] = estimate

    models: dict[str, PaceModel] = {}
    for circuit in circuits:
        fit = fits.get(circuit)
        at_circuit = laps.filter(pl.col("circuit") == circuit)
        local = _local_counts(at_circuit, compounds, settings.calibration.degradation_degree)
        # A coefficient this circuit could not estimate takes the pooled value, labelled with a
        # shrinkage of zero and with the laps this circuit actually ran -- never the pooled
        # lap count, which would claim evidence the circuit does not have.
        coefficients = {
            name: shrunk.get(circuit, {}).get(
                name,
                global_coefficients[name].model_copy(
                    update={"shrinkage": 0.0, "n_obs": local.get(name, at_circuit.height)}
                ),
            )
            for name in global_coefficients
        }
        models[circuit] = _model(
            coefficients,
            compounds,
            settings,
            circuit=circuit,
            residual_std_s=fit.residual_std_s if fit else pooled.residual_std_s,
            r_squared_within=fit.r_squared_within if fit else pooled.r_squared_within,
            n_laps=at_circuit.height,
            n_races=at_circuit.select("race_id").n_unique(),
            n_cars=at_circuit.select("car_id").n_unique(),
        )

    return CircuitPace(
        pooled=pooled,
        circuits=models,
        between_std=between_std,
        fitted=tuple(sorted(fits)),
        pooled_only=tuple(circuit for circuit in circuits if circuit not in fits),
    )


def _local_counts(
    at_circuit: pl.DataFrame, compounds: Sequence[str], degree: int
) -> dict[str, int]:
    """Laps behind each coefficient at one circuit: the compound's own laps where the coefficient
    belongs to a compound, and the circuit's total otherwise."""
    counts = {name: int(value) for name, value in at_circuit.group_by("compound").len().iter_rows()}
    local: dict[str, int] = {FUEL: at_circuit.height, DIRTY_AIR: at_circuit.height}
    for compound in compounds:
        laps_here = counts.get(compound, 0)
        local[compound_offset_term(compound)] = laps_here
        for power in range(1, degree + 1):
            local[degradation_term(compound, power)] = laps_here
    return local


def _pooled_coefficients(
    pooled: PaceModel, compounds: Sequence[str], settings: AppSettings
) -> dict[str, Estimate]:
    """The pooled model's coefficients, back in regressor-name form."""
    degree = settings.calibration.degradation_degree
    coefficients: dict[str, Estimate] = {FUEL: pooled.fuel_effect_s, DIRTY_AIR: pooled.dirty_air_s}
    for position, compound in enumerate(compounds):
        entry = pooled.compound(compound)
        for power in range(1, degree + 1):
            coefficients[degradation_term(compound, power)] = entry.coefficients[power - 1]
        if position > 0 and entry.offset_s is not None:
            coefficients[compound_offset_term(compound)] = entry.offset_s
    return coefficients
