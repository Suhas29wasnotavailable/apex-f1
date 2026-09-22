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
from pathlib import Path
from typing import Any

__all__ = ["TEMPLATE_PATH", "render_page", "write_page"]

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


def render_page(payload: dict[str, Any]) -> str:
    """Render the replay page with a payload embedded.

    Raises:
        FileNotFoundError: if the template is missing from the installed package.
    """
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    encoded = _escape_for_script(json.dumps(payload, separators=(",", ":")))
    title = str(payload.get("title", "Race replay"))
    return template.replace(_PAYLOAD_TOKEN, encoded).replace(_TITLE_TOKEN, title)


def write_page(payload: dict[str, Any], path: Path) -> Path:
    """Write the replay page to disk, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_page(payload), encoding="utf-8")
    return path
