#!/bin/sh
# End-to-end check on the target host, with the real models, the real STT
# backend and a throwaway MQTT broker. Nothing here touches Home Assistant.
#
#   LOCALAI_API_KEY=... STT_URL=http://gpu-box:8080/v1 sh tools/e2e.sh [recording.wav]
#
# Plays a recording through the built image (hearth:local) as if it were the
# microphone, then prints what reached MQTT and SQLite.
set -eu
SRC=$(cd "$(dirname "$0")/.." && pwd)
WAV=${1:-$SRC/tests/fixtures/cache/e2e.wav}
WORK=${E2E_DIR:-$HOME/hearth/e2e}
DOCKER=${DOCKER:-sudo -n docker}
SECONDS_TO_RUN=${E2E_SECONDS:-60}
rm -rf "$WORK"; mkdir -p "$WORK/data"
cp "$WAV" "$WORK/input.wav"
cat > "$WORK/hearth.toml" <<TOML
[audio]
source = "file"
file = "/e2e/input.wav"
[mqtt]
host = "127.0.0.1"
port = 18830
[api]
hosts = ["127.0.0.1"]
port = 18377
[summary]
hourly = false
nightly_at = ""
[stt.backends.localai]
base_url = "${STT_URL:?set STT_URL to the OpenAI-compatible endpoint}"
[llm.localai]
base_url = "${STT_URL}"
model = "${LLM_MODEL:-qwen3-4b}"
TOML
umask 077
printf 'LOCALAI_API_KEY=%s\nHEARTH_API_TOKENS=e2e:e2e-token\n' "$LOCALAI_API_KEY" > "$WORK/env"
cleanup() { $DOCKER rm -f hearth-e2e hearth-e2e-mqtt >/dev/null 2>&1 || true; rm -f "$WORK/env"; }
trap cleanup EXIT
cleanup
umask 022
$DOCKER run -d --name hearth-e2e-mqtt -p 127.0.0.1:18830:1883 eclipse-mosquitto:2 \
    mosquitto -c /mosquitto-no-auth.conf >/dev/null
sleep 2
$DOCKER exec hearth-e2e-mqtt mosquitto_sub -t '#' -v -W "$((SECONDS_TO_RUN + 60))" \
    -F '%I %t %p' > "$WORK/mqtt.log" 2>/dev/null &
printf 'LOCALAI_API_KEY=%s\nHEARTH_API_TOKENS=e2e:e2e-token\n' "$LOCALAI_API_KEY" > "$WORK/env"; chmod 600 "$WORK/env"
$DOCKER run -d --name hearth-e2e --network host --user "$(id -u):$(getent group audio | cut -d: -f3)" \
    --env-file "$WORK/env" -e TZ=Europe/London \
    -v "$WORK/hearth.toml:/config/hearth.toml:ro" -v "$WORK:/e2e:ro" -v "$WORK/data:/data" hearth:local >/dev/null
sleep 12
echo "== cpu while listening (three samples)"
for i in 1 2 3; do $DOCKER stats --no-stream --format 'cpu={{.CPUPerc}} mem={{.MemUsage}}' hearth-e2e; sleep 3; done
sleep "$((SECONDS_TO_RUN - 30))"
echo "== container"; $DOCKER inspect -f '{{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{end}}' hearth-e2e
echo "== summary of what was transcribed (real LLM)"
curl -s -m 280 -X POST -H 'Authorization: Bearer e2e-token' -H 'Content-Type: application/json' \
    -d '{"kind":"manual","hours":1}' http://127.0.0.1:18377/api/v1/summaries/run || echo "summary call failed"
echo; echo "== status"; curl -s -m 5 -H 'Authorization: Bearer e2e-token' http://127.0.0.1:18377/api/v1/status; echo
sleep 2
echo "== log"; $DOCKER logs hearth-e2e 2>&1 | grep -v -E 'XNNPACK|cpuid' | tail -25
echo "== mqtt (state and events, in order)"
grep -v 'homeassistant/' "$WORK/mqtt.log" | grep -v -E 'hearth/(stt|mic)/' | cut -c1-200
echo "== mqtt discovery topics: $(grep -c 'homeassistant/.*/config' "$WORK/mqtt.log")"
echo "== transcripts"
python3 - "$WORK/data/hearth.sqlite" <<'PY'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
for r in db.execute("select started_at, backend, text from segments order by started_at"):
    print(" ", r[0][11:19], r[1], "|", r[2])
print("  sound events:", db.execute("select category, score from sound_events").fetchall())
PY
echo "== files written"; (cd "$WORK/data" && find . -type f | sort)
