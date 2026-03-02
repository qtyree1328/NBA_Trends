#!/bin/bash

# Run fetch_nba_data.py with window=5 and window=7, then commit and push

set -euo pipefail

cd "$(dirname "$0")"

# Activate conda so we get the right Python with nba_api, pandas, etc.
if [[ ! -x /opt/miniconda3/bin/conda ]]; then
    echo "[FAIL] Conda not found at /opt/miniconda3/bin/conda"
    exit 1
fi
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
conda activate base

MAX_RETRIES="${MAX_RETRIES:-5}"
NBA_API_TIMEOUT="${NBA_API_TIMEOUT:-180}"
SLEEP_BETWEEN_CALLS_SEC="${SLEEP_BETWEEN_CALLS_SEC:-1.25}"
MIN_TEAMS_REQUIRED="${MIN_TEAMS_REQUIRED:-25}"

python - <<'PY'
from importlib.util import find_spec
required = ["numpy", "pandas", "tqdm", "nba_api"]
missing = [pkg for pkg in required if find_spec(pkg) is None]
if missing:
    raise SystemExit(f"[FAIL] Missing Python packages in active env: {', '.join(missing)}")
print("[OK] Python environment ready")
PY

validate_output() {
    local file="$1"
    local min_teams="$2"
    python - "$file" "$min_teams" <<'PY'
import json
import sys
from pathlib import Path

file_path = Path(sys.argv[1])
min_teams = int(sys.argv[2])

if not file_path.exists():
    raise SystemExit(f"[FAIL] Missing output file: {file_path}")

data = json.loads(file_path.read_text())
teams = data.get("teams", [])
if len(teams) < min_teams:
    raise SystemExit(
        f"[FAIL] {file_path.name} has {len(teams)} teams; expected at least {min_teams}."
    )

print(f"[OK] {file_path.name}: {len(teams)} teams")
PY
}

run_window() {
    local window="$1"
    local outfile="nba_trends_data_window${window}.json"
    echo "=== Running with WINDOW=${window} ==="
    WINDOW="${window}" \
    MAX_RETRIES="${MAX_RETRIES}" \
    NBA_API_TIMEOUT="${NBA_API_TIMEOUT}" \
    SLEEP_BETWEEN_CALLS_SEC="${SLEEP_BETWEEN_CALLS_SEC}" \
    MIN_TEAMS_REQUIRED="${MIN_TEAMS_REQUIRED}" \
    python fetch_nba_data.py
    validate_output "${outfile}" "${MIN_TEAMS_REQUIRED}"
}

run_window 5

echo ""
run_window 7

echo ""
echo "=== Committing and pushing ==="
git add .
git diff --cached --quiet || git commit -m 'update'
git push

echo ""
echo "=== Done ==="
