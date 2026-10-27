#!/usr/bin/env bash
# Start the overnight pilot (PLAN §9.2) in a detached screen session, under caffeinate so the Mac stays awake.
#   bash tools/night.sh <night_id> [more mmpsi.pilot arguments]      e.g. bash tools/night.sh 2026-10-27_n1
# Progress: data/logs/<night_id>/heartbeat.json (rewritten every minute); attach: screen -r pilot-<night_id>
set -euo pipefail
night="${1:?usage: bash tools/night.sh <night_id>}"
shift
cd "$(dirname "$0")/.."
logs="data/logs/$night"
mkdir -p "$logs"
# `screen -ls` exits with 1 even when it lists sessions, which pipefail would turn into a failed pipeline
running() { { screen -ls 2>/dev/null || true; } | grep -q "[.]pilot-$night[[:space:]]"; }
if running; then
  echo "screen session pilot-$night is already running"; exit 1
fi
pmset -g batt | grep -q "AC Power" || echo "warning: the Mac is on battery; plug it in before leaving it overnight"
# caffeinate -i: no idle sleep; -m: no disk sleep; -s: no system sleep while on AC power. Closing the lid still
# sleeps a MacBook without an external display, so the lid stays open. After the run, the reports are rendered once
# more from facts.json with the current report code, so report fixes made during the night are included.
screen -dmS "pilot-$night" bash -c "caffeinate -ims .venv/bin/python -m mmpsi.pilot --config configs/pilot_n1.yaml \
  --night '$night' $* 2>&1 | tee -a '$logs/night.out'; echo \"exit code \${PIPESTATUS[0]}\" >> '$logs/night.out'; \
  .venv/bin/python -m mmpsi.pilot --night '$night' --report-only >> '$logs/night.out' 2>&1"
sleep 1
running || { echo "the screen session did not start; see $logs/night.out"; exit 1; }
echo "started screen session pilot-$night"
echo "  stage:   cat $logs/heartbeat.json"
echo "  log:     tail -f $logs/pilot.log"
echo "  attach:  screen -r pilot-$night   (detach again with Ctrl-a d)"
echo "  reports: reports/$night/"
