"""Shared test fixtures.

Tests must never depend on the developer's environment or on network access. CI runs with no
FastF1 credentials and no internet, so anything touching real data belongs behind the
``network`` marker and is fed by committed fixtures instead.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from apex.config import get_settings


@pytest.fixture(scope="session", autouse=True)
def _isolate_apex_environment() -> Iterator[None]:
    """Remove any ``APEX_`` variables so a developer's shell cannot change test outcomes."""
    with pytest.MonkeyPatch.context() as patch:
        for key in [key for key in os.environ if key.startswith("APEX_")]:
            patch.delenv(key, raising=False)
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()
