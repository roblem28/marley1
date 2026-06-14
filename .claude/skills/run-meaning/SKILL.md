---
name: run-meaning
description: Run, smoke-test, and drive the Marley1 meaning/ module (MTP v0.1 "meaning packet" message type + ed25519 node identity) across the Fat Man + Little Boy mesh. Use to launch the receiver, send a test message, verify the encode -> sign -> send -> verify -> render round-trip, or inspect the [ALERTA]/[INFO]/[RUTINA] Spanish rendering. Triggers: run meaning, start meaning module, smoke test meaning, send a meaning packet, screenshot/inspect the render.
---

# run-meaning — drive the MTP v0.1 meaning-packet mesh

`meaning/` is a **two-node distributed message type**, not a local app:

- **Fat Man** (`pirat28@100.97.87.86`, Jetson) — *sender*: `node.py send "text"` runs
  `encode()` (local llama-server `:8080`) → `sign()` (ed25519) → POSTs the packet.
- **Little Boy** (`marley1@100.110.181.128`, Pi 5) — *receiver*: `node.py serve` (systemd
  unit `meaning-render`, Flask `:8082`) verifies the signature against the pinned peer and
  `render()`s Spanish via llama-server `:8081`, prefixed `[ALERTA]`/`[INFO]`/`[RUTINA]`.

You **cannot** drive this with a single binary — there is no GUI and no single host. The
driver is **`smoke.sh`**, which exercises the whole round-trip and asserts HTTP 200 + the
expected prefix + a grown receiver inbox. It runs from either:

- a **control host** (workstation) with key-based SSH to both nodes, or
- **on Fat Man** (the sender) — the local encode/send step runs directly, the receiver over
  Tailscale SSH.

Both were verified this session. Note: a node cannot SSH to *itself*, which is why the
driver runs the local side without SSH instead of looping back.

> Paths below are relative to the unit dir `~/marley1/`. The driver lives at
> `.claude/skills/run-meaning/smoke.sh`.

## Prerequisites

- **Tailscale SSH** to both nodes with key auth (the `100.x` addresses above). LAN
  `192.168.88.x` does **not** work for TCP — see Gotchas.
- The services must be up. They are **systemd-managed** and survive reboot, so normally
  nothing to do. To confirm:

```bash
ssh pirat28@100.97.87.86   'systemctl --user is-active meaning-llama'   # sender LLM :8080
ssh marley1@100.110.181.128 'systemctl is-active meaning-llama meaning-render'  # :8081, :8082
```

- Python deps (already installed on both nodes). Only if rebuilding a node from scratch
  (Little Boy is Debian Bookworm → `--break-system-packages` is required):

```bash
ssh marley1@100.110.181.128 'pip3 install --break-system-packages pynacl flask requests'
```

## Run (agent path) — the driver

```bash
bash .claude/skills/run-meaning/smoke.sh
```

Expected tail on success (takes ~30–60s; the encode + render are CPU-only LLM calls):

```
== 4. assertions ==
  PASS: 200 + \[ALERTA\] + signed + inbox grew (11 -> 12)
  inbox<- 2026-06-14T15:56:49...  c847f775-...  [ALERTA] Por precaución ante el aviso de tormenta, ...
ALL_GOOD
```

Send a custom message (the prefix tracks the encoder's priority, so override the assertion
when you don't expect an emergency):

```bash
MEANING_EXPECT_PREFIX='\[INFO\]|\[RUTINA\]|\[ALERTA\]' \
  bash .claude/skills/run-meaning/smoke.sh "Reminder: toolbox talk at 9am in the site office."
```

Env overrides: `MEANING_SENDER`, `MEANING_RECEIVER`, `MEANING_NODE_PY`, `MEANING_INBOX`,
`MEANING_EXPECT_PREFIX`.

Inspect what the receiver rendered (it appends every accepted packet):

```bash
ssh marley1@100.110.181.128 'tail -3 ~/meaninglayer/inbox.log'
ssh marley1@100.110.181.128 'sudo journalctl -u meaning-render -n 20 --no-pager'
```

## Direct invocation (one node at a time)

Sender only — encode + sign + send, prints byte sizes + the receiver's JSON reply:

```bash
ssh pirat28@100.97.87.86 'python3 ~/marley1/meaning/node.py send "Storm coming this afternoon. Move all equipment off the slab and cover open trenches by 3pm."'
```

The `packet.py` API (`encode(text,src_lang)`, `sign(pkt)`, `verify(pkt,peer_pubkey)`,
`render(pkt,target_lang)`, `sizes(text,pkt)`) is importable for testing internals:

```bash
ssh pirat28@100.97.87.86 'cd ~/marley1/meaning && python3 -c "import packet; print(packet.node_fingerprint())"'
```

## Human path

There isn't a useful one — the receiver is a headless Flask service under systemd and the
sender is a CLI. Manual `node.py serve` in a terminal works but the systemd unit already
owns `:8082`; stop it first (`sudo systemctl stop meaning-render`) or you'll get
`Address already in use`.

## Gotchas (battle scars — these cost real time)

- **Jetson GPU is a trap for this model.** llama-server on Fat Man must run **CPU-only via
  `-dev none`**. `-ngl 0` and `CUDA_VISIBLE_DEVICES=` do **not** force CPU — the Orin's
  *integrated* GPU is always CUDA device 0, and the auto-`-fit` logic puts tensors on it
  anyway. It then dies with `NvMapMemAllocInternalTagged ... error 12` / `cudaMalloc failed`
  because GPU memory is shared with system RAM that's full of page cache (no sudo to drop
  caches). `-dev none` is the only thing that worked.
- **LAN TCP between the nodes is firewalled.** Little Boy's `ufw` policy is DROP with only
  `tailscale0` allowed, so `192.168.88.14:8082` (and `:22`) **time out** even though ICMP
  ping succeeds. The sender POSTs to the **Tailscale** IP `100.110.181.128:8082`
  (`MEANING_PEER_URL` default in `packet.py`). Use `100.x` for everything.
- **The two nodes use different LLM ports.** Sender encodes on `:8080`, receiver renders on
  `:8081`. The receiver's systemd unit sets `Environment=MEANING_LLAMA_URL=...:8081`; the
  sender uses the `:8080` default. Don't "fix" them to match.
- **`systemctl --user` on Fat Man needs a runtime dir over SSH.** The sender's llama-server
  is a **user** unit (no sudo on the Jetson) with linger enabled. Non-login SSH has no bus:
  prefix with `export XDG_RUNTIME_DIR=/run/user/$(id -u)` before any `systemctl --user`.
- **SSH to the nodes drops on long commands.** Anything that blocks >~20s (model load,
  the 40-sentence encode) tends to return exit 255 mid-stream. Launch long work detached
  (`setsid ... </dev/null &`) and poll, and add `-o ServerAliveInterval=10`. The driver
  keeps each SSH call short for this reason.
- **Identity is reused, never regenerated.** Keys live at `~/.meaninglayer/key` (sender) and
  the pinned peer at `~/.meaninglayer/peers.json` (receiver), **outside** the repo. Fat Man's
  fingerprint is `cc5315d880d69787`. `packet._identity()` raises if the key is missing rather
  than minting a new one — that's intentional.
- **Priority drives the prefix, not intent.** `[ALERTA]` requires the encoder to classify
  `pri=emergency`. The storm test message reliably does; calmer messages come back
  `info`/`routine` → `[INFO]`/`[RUTINA]`. Set `MEANING_EXPECT_PREFIX` accordingly.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FAIL: sender llama :8080 down` | `ssh pirat28@100.97.87.86 'export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user restart meaning-llama'`; wait ~20s for model load. |
| `FAIL: meaning-render not active` | `ssh marley1@100.110.181.128 'sudo systemctl restart meaning-render'`; check `journalctl -u meaning-render`. |
| Send hangs then `ConnectTimeout ... 192.168.88.x` | You're using a LAN IP. Use the `100.x` Tailscale address. |
| `node identity key missing: ~/.meaninglayer/key` | The signing key is gone — do **not** regenerate blindly (it's the node identity and the peer pins its pubkey). Restore from backup. |
| Render service won't bind (`Address already in use`) | A manual `node.py serve` is holding `:8082`; the systemd unit owns it. `fuser -k 8082/tcp` then `sudo systemctl restart meaning-render`. |
