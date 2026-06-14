#!/usr/bin/env bash
# Drive the Meaning Layer (MTP v0.1) two-node mesh end-to-end and assert the full
# encode -> ed25519-sign -> POST -> verify -> render round-trip works, producing a
# priority-prefixed rendering.
#
# Default (forward leg):  Fat Man (en) -> Little Boy renders Spanish  [ALERTA]/[INFO]/[RUTINA]
# REVERSE=1 (reverse leg): Little Boy (es) -> Fat Man renders English [ALERT]/[INFO]/[ROUTINE]
#
# The mesh is symmetric: each node pins the OTHER's ed25519 pubkey in
# ~/.meaninglayer/peers.json and runs a receiver on :8082 (Little Boy = system unit
# rendering Spanish via :8081; Fat Man = user unit rendering English via :8080).
#
# Runs from ANY host that can reach both nodes (control host, or on a node itself --
# a node can't SSH to itself, so the local side runs without SSH).
#
# Usage:  ./smoke.sh ["custom message"]            # forward leg
#         REVERSE=1 ./smoke.sh ["custom reply"]    # reverse leg
# Exit 0 = PASS (status 200 + expected prefix + ed25519 sig + receiver inbox grew).
set -uo pipefail

FATMAN="${MEANING_FATMAN:-pirat28@100.97.87.86}"          # encoder llama :8080, English receiver
LITTLEBOY="${MEANING_LITTLEBOY:-marley1@100.110.181.128}" # render llama  :8081, Spanish receiver
FM_NODE_PY="${MEANING_FM_NODE_PY:-/home/pirat28/marley1/meaning/node.py}"
LB_NODE_PY="${MEANING_LB_NODE_PY:-/home/marley1/marley1/meaning/node.py}"
FM_INBOX="${MEANING_FM_INBOX:-/home/pirat28/meaninglayer/inbox.log}"
LB_INBOX="${MEANING_LB_INBOX:-/home/marley1/meaninglayer/inbox.log}"
REVERSE="${REVERSE:-0}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=40"

fail() { echo "FAIL: $1"; exit 1; }

# run a command "at" a host: locally (no SSH) if that host is this machine, else over SSH.
_local_ips() { hostname -I 2>/dev/null; command -v tailscale >/dev/null 2>&1 && tailscale ip -4 2>/dev/null; }
is_local() { _local_ips | tr ' ' '\n' | grep -qx "${1#*@}"; }
run_at() { if is_local "$1"; then eval "$2"; else $SSH "$1" "$2"; fi; }

if [ "$REVERSE" = 1 ]; then
  LEG="REVERSE (Little Boy es -> Fat Man en)"
  SENDER="$LITTLEBOY";  RECEIVER="$FATMAN"
  SEND_ENC_PORT=8081;   RECV_RENDER_PORT=8080
  NODE_PY="$LB_NODE_PY"; INBOX="$FM_INBOX"
  MSG="${1:-Ya voy, cinco minutos.}"
  EXPECT_PREFIX="${MEANING_EXPECT_PREFIX:-\[ROUTINE\]}"           # calm reply => routine => [ROUTINE]
  # receiver is Fat Man's *user* unit -> needs the runtime dir over non-login SSH
  RENDER_CHECK='export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user is-active meaning-render'
  # Little Boy encodes on :8081 and POSTs to Fat Man's receiver; src_lang = es
  SEND_CMD="MEANING_LLAMA_URL=http://127.0.0.1:8081/v1/chat/completions MEANING_PEER_URL=http://100.97.87.86:8082/packet python3 $NODE_PY send \"$MSG\" es"
else
  LEG="FORWARD (Fat Man en -> Little Boy es)"
  SENDER="$FATMAN";     RECEIVER="$LITTLEBOY"
  SEND_ENC_PORT=8080;   RECV_RENDER_PORT=8081
  NODE_PY="$FM_NODE_PY"; INBOX="$LB_INBOX"
  MSG="${1:-Storm coming this afternoon. Move all equipment off the slab and cover open trenches by 3pm.}"
  EXPECT_PREFIX="${MEANING_EXPECT_PREFIX:-\[ALERTA\]}"            # storm => emergency => [ALERTA]
  # receiver is Little Boy's *system* unit
  RENDER_CHECK='systemctl is-active meaning-render'
  # Fat Man encodes on :8080 and POSTs to Little Boy (PEER_URL default); src_lang = en
  SEND_CMD="python3 $NODE_PY send \"$MSG\""
fi

echo "== leg: $LEG =="

echo "== 1. health =="
run_at "$SENDER"   "curl -sf -m6 http://127.0.0.1:$SEND_ENC_PORT/v1/models >/dev/null" \
  && echo "  sender llama :$SEND_ENC_PORT OK"   || fail "sender llama :$SEND_ENC_PORT down"
run_at "$RECEIVER" "curl -sf -m6 http://127.0.0.1:$RECV_RENDER_PORT/v1/models >/dev/null" \
  && echo "  receiver llama :$RECV_RENDER_PORT OK" || fail "receiver llama :$RECV_RENDER_PORT down"
RSTATE=$(run_at "$RECEIVER" "$RENDER_CHECK")
echo "  meaning-render: $RSTATE"
[ "$RSTATE" = active ] || fail "meaning-render not active"

echo "== 2. inbox baseline =="
BEFORE=$(run_at "$RECEIVER" "wc -l < $INBOX 2>/dev/null || echo 0")
echo "  inbox lines: $BEFORE"

echo "== 3. encode + sign + send (from sender) =="
OUT=$(run_at "$SENDER" "$SEND_CMD") || fail "node.py send errored"
echo "$OUT" | sed 's/^/  /'

echo "== 4. assertions =="
echo "$OUT" | grep -q "status 200"            || fail "no HTTP 200 from receiver"
echo "$OUT" | grep -qE "$EXPECT_PREFIX"       || fail "expected prefix $EXPECT_PREFIX not in rendering"
echo "$OUT" | grep -qE "from\): [0-9a-f]{16}" || fail "no ed25519 node fingerprint in output"
AFTER=$(run_at "$RECEIVER" "wc -l < $INBOX 2>/dev/null || echo 0")
[ "$AFTER" -gt "$BEFORE" ] || fail "receiver inbox did not grow ($BEFORE -> $AFTER)"
echo "  PASS: 200 + $EXPECT_PREFIX + signed + inbox grew ($BEFORE -> $AFTER)"
run_at "$RECEIVER" "tail -1 $INBOX" | sed 's/^/  inbox<- /'
echo "ALL_GOOD"
