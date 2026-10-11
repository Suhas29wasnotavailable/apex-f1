#!/usr/bin/env bash
# Build the season replay and publish it to GitHub Pages.
#
# The page embeds simulated race data, which is never committed (see the large-files hook), and it
# needs params/params.json and the traced tracks, which are generated artefacts not in the
# repository, so CI cannot build it. Instead the page is built here, attached to the `replay`
# release, and the Pages workflow deploys that file. The build is deterministic, so republishing
# from unchanged parameters serves the same bytes.
set -euo pipefail

tag="replay"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT

source_rev="$(git rev-parse --short HEAD)"
uv run --extra data --extra fit apex replay --no-open --out "$build/index.html"

if ! gh release view "$tag" >/dev/null 2>&1; then
  gh release create "$tag" --title "Season replay" --latest=false \
    --notes "The published replay page, rebuilt by \`make publish\`. Served by GitHub Pages."
fi
gh release upload "$tag" "$build/index.html" --clobber
gh release edit "$tag" --notes "The published replay page, built from ${source_rev} by \`make publish\`. Served by GitHub Pages."
gh workflow run pages.yml
echo "Uploaded the replay built from ${source_rev}; the Pages workflow is deploying it."
