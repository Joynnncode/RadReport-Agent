#!/usr/bin/env bash
# Run the fast test suite against ONLY the files git would ship.
#
# Why this exists. The first push to GitHub failed CI with six errors that all
# passed locally, because six tests depended on files that are gitignored:
# evals/gold_set.jsonl and data/reports.csv. My working tree had them. A fresh
# clone does not. "Works on my machine" in its purest form.
#
# This copies exactly the tracked+untracked-but-not-ignored set into a temp dir
# and runs the fast suite there, so the failure surfaces in ten seconds instead
# of after a push.
#
#   ./scripts/check_fresh_clone.sh
#
# Run it before every push.

set -euo pipefail
cd "$(dirname "$0")/.."

REPO="$PWD"
PY="${PYTHON:-$REPO/.venv/bin/python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Staging the files git would ship into $TMP ..."
git ls-files --cached --others --exclude-standard | while read -r f; do
  mkdir -p "$TMP/$(dirname "$f")"
  cp "$f" "$TMP/$f"
done

count=$(find "$TMP" -type f | wc -l | tr -d ' ')
echo "  $count file(s)"
echo "  gold set : $([ -f "$TMP/evals/gold_set.jsonl" ] && echo present || echo ABSENT)"
echo "  corpus   : $([ -f "$TMP/data/reports.csv" ] && echo present || echo 'absent (expected)')"
echo

cd "$TMP"
"$PY" -m pytest -m "not slow and not network" -q
