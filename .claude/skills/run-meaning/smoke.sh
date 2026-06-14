#!/usr/bin/env bash
# Drive the Meaning Layer (MTP v0.1) two-node mesh end-to-end and assert the
# full encode -> ed25519-sign -> POST -> verify -> render round-trip works,
# producing a priority-prefixed Spanish rendering.
#
# Runs from ANY host that can reach both nodes:
#   - a control host (workstation) with key SSH to both, OR
#   - on a node itself (the local node's step runs directly; the peer over SSH).
# Nodes can't SSH to themselves, so the script runs the local side without SSH.
#
# Usage:  ./smoke.sh ["custom english message"]
# Exit 0 = PASS (status 200 + expected prefix + ed25519 sig + receiver inbox grew).
set -uo pipefail

SENDER="${MEANING_SENDER:-pirat28@100.97.87.86}"          # Fat Man  (encoder)
RECEIVER="${MEANING_RECEIVER:-marley1@100.110.181.128}"   # Little Boy (renderer)
NODE_PY="${MEANING_NODE_PY:-/home/pirat28/marley1/meaning/node.py}"
INBOX="${MEANING_INBOX:-/home/marley1/meaninglayer/inbox.log}"
MSG="${1:-Storm coming this afternoon. Move all equipment off the slab and cover open trenches by 3pm.}"
EXPECT_PREFIX="${MEANING_EXPECT_PREFIX:-\[ALERTA\]}"      # storm => emergency => [ALERTA]
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=40"

fail() { echo "FAIL: $1"; exit 1; }

# run a command "at" a host: locally (no SSH) if that host is this machine, else over SSH.
_local_ips() { hostname -I 2>/dev/null; command -v tailscale >/dev/null 2>&1 && tailscale ip -4 2>/dev/null; }
is_local() { _local_ips | tr ' ' '\n' | grep -qx "${1#*@}"; }
run_at() { if is_local "$1"; then eval "$2"; else $SSH "$1" "$2"; fi; }

echo "== 1. health =="
run_at "$SENDER"   'curl -sf -m6 http://127.0.0.1:8080/v1/models >/dev/null' \
  && echo "  sender llama :8080 OK"   || fail "sender llama :8080 down"
run_at "$RECEIVER" 'curl -sf -m6 http://127.0.0.1:8081/v1/models >/dev/null' \
  && echo "  receiver llama :8081 OK" || fail "receiver llama :8081 down"
RSTATE=$(run_at "$RECEIVER" 'systemctl is-active meaning-render')
echo "  meaning-render: $RSTATE"
[ "$RSTATE" = active ] || fail "meaning-render not active"

echo "== 2. inbox baseline =="
BEFORE=$(run_at "$RECEIVER" "wc -l < $INBOX 2>/dev/null || echo 0")
echo "  inbox lines: $BEFORE"

echo "== 3. encode + sign + send (from sender) =="
OUT=$(run_at "$SENDER" "python3 $NODE_PY send \"$MSG\"") || fail "node.py send errored"
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
