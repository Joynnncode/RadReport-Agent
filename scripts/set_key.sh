#!/usr/bin/env bash
# Write an API key into .env without it appearing in your shell history.
#
#   ./scripts/set_key.sh GEMINI_API_KEY
#   ./scripts/set_key.sh GROQ_API_KEY
#
# Prompts for the value with echo off, so nothing is displayed or recorded.
# Replaces the line if the variable already exists, appends it otherwise.

set -euo pipefail

VAR="${1:-}"
if [[ -z "$VAR" ]]; then
  echo "usage: $0 <VARIABLE_NAME>" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
[[ -f .env ]] || cp .env.example .env

printf 'Paste value for %s (input hidden): ' "$VAR" >&2
read -rs VALUE
echo >&2

# Strip accidental surrounding quotes and whitespace -- the most common paste error.
VALUE="$(printf '%s' "$VALUE" | tr -d '[:space:]' | sed -E "s/^['\"]//; s/['\"]$//")"

if [[ -z "$VALUE" ]]; then
  echo "Nothing pasted; .env unchanged." >&2
  exit 1
fi

if grep -q "^${VAR}=" .env; then
  # Write via a temp file: sed -i '' with a secret in the pattern can leak via ps.
  awk -v var="$VAR" -v val="$VALUE" \
      'BEGIN{FS=OFS="="} $1==var {print var "=" val; next} {print}' .env > .env.tmp
  mv .env.tmp .env
else
  printf '%s=%s\n' "$VAR" "$VALUE" >> .env
fi

chmod 600 .env
echo "Set ${VAR} (${#VALUE} chars) in .env" >&2
