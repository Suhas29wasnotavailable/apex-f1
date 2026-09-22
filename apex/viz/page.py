"""Writing the replay out as one file you can open.

The page is self-contained on purpose. A race replay that needs a server running is a race replay
nobody watches: this writes a single HTML file with the track, the flags and every car's crossing
times embedded in it, so it opens from the filesystem, survives being emailed, and works offline.
P5's serving layer will stream races live; this is the artefact you can keep.

The markup lives in ``templates/replay.html`` rather than in a Python string, so the front end can
be edited as HTML with a browser's own tooling, and all this module does is substitute the payload
into it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["TEMPLATE_PATH", "render_page", "season_payload", "write_page"]

TEMPLATE_PATH = Path(__file__).parent / "templates" / "replay.html"

_PAYLOAD_TOKEN = "__PAYLOAD__"
_TITLE_TOKEN = "__TITLE__"


def _escape_for_script(payload: str) -> str:
    """Make JSON safe to embed inside a ``<script>`` element.

    A literal ``</script>`` anywhere in the data would close the element early and put the rest of
    the race into the document as markup. Driver codes and circuit names make that vanishingly
    unlikely, but the failure is silent and total, so the sequence is escaped rather than trusted.
    """
    return payload.replace("</", "<\\/")


def season_payload(races: Sequence[dict[str, Any]], *, title: str = "") -> dict[str, Any]:
    """Bundle several simulated races into one page.

    One page holding a calendar is worth more than a folder of pages holding one race each: the
    interesting question is how differently the same twenty cars race at Monaco and at Monza, and
    that only reads when you can switch between them without losing your place.

    Raises:
        ValueError: if there are no races to show.
    """
    if not races:
        raise ValueError("a replay needs at least one race")
    return {
        "title": title or "APEX race replays",
        "races": list(races),
    }


def render_page(payload: dict[str, Any]) -> str:
    """Render the replay page with a payload embedded.

    Raises:
        FileNotFoundError: if the template is missing from the installed package.
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    # A single race is wrapped so the page only ever has one shape to read.
    bundle = (
        payload if "races" in payload else season_payload([payload], title=payload.get("title", ""))
    )
    encoded = _escape_for_script(json.dumps(bundle, separators=(",", ":")))
    title = str(bundle.get("title", "Race replay"))
    return template.replace(_PAYLOAD_TOKEN, encoded).replace(_TITLE_TOKEN, title)


def write_page(payload: dict[str, Any], path: Path) -> Path:
    """Write the replay page to disk, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_page(payload), encoding="utf-8")
    return path
