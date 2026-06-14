---
name: run-meaning
description: Run, smoke-test, and drive the Marley1 meaning/ module (MTP v0.1 "meaning packet" message type + ed25519 node identity) across the Fat Man + Little Boy mesh. The mesh is SYMMETRIC: a forward leg (Fat Man -> Little Boy, Spanish [ALERTA]/[INFO]/[RUTINA]) and a reverse leg (Little Boy -> Fat Man, English [ALERT]/[INFO]/[ROUTINE]). Use to launch a receiver, send a test message either direction, verify the encode -> sign -> send -> verify -> render round-trip, or inspect the rendering. Triggers: run meaning, start meaning module, smoke test meaning, reverse leg, round-trip, send a meaning packet, screenshot/inspect the render.
---

# run-meaning — drive the MTP v0.1 meaning-packet mesh

`meaning/` is a **two-node distributed message type**, not a local app. The mesh is
**symmetric** — each node both *sends* (encode + sign + POST) and *receives* (verify +
render on `:8082`), and each pins the **other's** ed25519 pubkey in
`~/.meaninglayer/peers.json`, verifying every packet before it is rendered:

- **Fat Man** (`pirat28@100.97.87.86`, Jetson) — encoder llama `:8080`; *receiver* is the
  **user** unit `meaning-render` (Flask `:8082`) that renders **English** via `:8080`,
  prefixed `[ALERT]`/`[INFO]`/`[ROUTINE]`. Pins Little Boy (`cb9246626aed27bf`).
- **Little Boy** (`marley1@100.110.181.128`, Pi 5) — render llama `:8081`; *receiver* is the
  **system** unit `meaning-render` (Flask `:8082`) that renders **Spanish** via `:8081`,
  prefixed `[ALERTA]`/`[INFO]`/`[RUTINA]`. Pins Fat Man (`cc5315d880d69787`).

So the round-trip runs both ways:

- **Forward leg** (default): Fat Man encodes English → Little Boy renders Spanish.
- **Reverse leg** (`REVERSE=1`): Little Boy encodes Spanish → Fat Man renders English.

You **cannot** drive this with a single binary — there is no GUI and no single host. The
driver is **`smoke.sh`**, which exercises a full round-trip and asserts HTTP 200 + the
expected prefix + a grown receiver inbox. It runs from either:

- a **control host** (workstation) with key-based SSH to both nodes, or
- **on a node** — the local encode/send step runs directly, the peer over Tailscale SSH.

Note: a node cannot SSH to *itself*, which is why the driver runs the local side without SSH.

> Paths below are relative to the unit dir `~/marley1/`. The driver lives at
> `.claude/skills/run-meaning/smoke.sh`.

## Prerequisites

- **Tailscale SSH** to both nodes with key auth (the `100.x` addresses above). LAN
  `192.168.88.x` does **not** work for TCP — see Gotchas.
- The services must be up. They are **service-managed** and survive reboot (Fat Man via
  user units + linger; Little Boy via system units), so normally nothing to do. To confirm:

```bash
ssh pirat28@100.97.87.86   'export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user is-active meaning-llama meaning-render'  # encoder :8080 + English receiver :8082
ssh marley1@100.110.181.128 'systemctl is-active meaning-llama meaning-render'  # render :8081 + Spanish receiver :8082
```

- Python deps (already installed on both nodes). Only if rebuilding a node from scratch
  (Little Boy is Debian Bookworm → `--break-system-packages` is required):

```bash
ssh marley1@100.110.181.128 'pip3 install --break-system-packages pynacl flask requests'
```

## Run (agent path) — the driver

Forward leg (default — Fat Man → Little Boy, Spanish):

```bash
bash .claude/skills/run-meaning/smoke.sh
```

Reverse leg (Little Boy → Fat Man, English) — behind the `REVERSE=1` flag:

```bash
REVERSE=1 bash .claude/skills/run-meaning/smoke.sh
```

`REVERSE=1` flips every role: Little Boy encodes on `:8081`, signs with its own identity,
and POSTs to Fat Man's `:8082` receiver, which verifies against the pinned Little Boy pubkey
and renders **English** (`[ALERT]`/`[INFO]`/`[ROUTINE]`). The forward test is unchanged and
remains the default.

Expected tail on success (takes ~30–60s; encode + render are CPU-only LLM calls):

```
== 4. assertions ==
  PASS: 200 + \[ALERTA\] + signed + inbox grew (14 -> 15)
  inbox<- 2026-06-14T16:22:50...  87d89946-...  [ALERTA] Por precaución ante el aviso de tormenta, ...
ALL_GOOD
```

Send a custom message (the prefix tracks the encoder's priority, so override the assertion
when you don't expect the default priority):

```bash
MEANING_EXPECT_PREFIX='\[INFO\]|\[RUTINA\]|\[ALERTA\]' \
  bash .claude/skills/run-meaning/smoke.sh "Reminder: toolbox talk at 9am in the site office."

REVERSE=1 MEANING_EXPECT_PREFIX='\[INFO\]|\[ROUTINE\]|\[ALERT\]' \
  bash .claude/skills/run-meaning/smoke.sh "Aviso: hay una fuga de gas en el sótano, evacuen ahora."
```

Env overrides: `REVERSE`, `MEANING_FATMAN`, `MEANING_LITTLEBOY`, `MEANING_FM_NODE_PY`,
`MEANING_LB_NODE_PY`, `MEANING_FM_INBOX`, `MEANING_LB_INBOX`, `MEANING_EXPECT_PREFIX`.

Inspect what a receiver rendered (each appends every accepted packet):

```bash
ssh marley1@100.110.181.128 'tail -3 ~/meaninglayer/inbox.log'   # Spanish renders (forward)
ssh pirat28@100.97.87.86   'tail -3 ~/meaninglayer/inbox.log'    # English renders (reverse)
ssh marley1@100.110.181.128 'sudo journalctl -u meaning-render -n 20 --no-pager'
ssh pirat28@100.97.87.86   'export XDG_RUNTIME_DIR=/run/user/$(id -u); journalctl --user -u meaning-render -n 20 --no-pager'
```

## Direct invocation (one node at a time)

Forward — Fat Man encodes + signs + sends to Little Boy:

```bash
ssh pirat28@100.97.87.86 'python3 ~/marley1/meaning/node.py send "Storm coming this afternoon. Move all equipment off the slab and cover open trenches by 3pm."'
```

Reverse — Little Boy encodes (on `:8081`) + signs + sends to Fat Man (`:8082`):

```bash
ssh marley1@100.110.181.128 'MEANING_LLAMA_URL=http://127.0.0.1:8081/v1/chat/completions \
  MEANING_PEER_URL=http://100.97.87.86:8082/packet \
  python3 ~/marley1/meaning/node.py send "Ya voy, cinco minutos." es'
```

The `packet.py` API (`encode(text,src_lang)`, `sign(pkt)`, `verify(pkt,peer_pubkey)`,
`render(pkt,target_lang)`, `sizes(text,pkt)`) is importable for testing internals. The
receiver's render language is set by `MEANING_RENDER_LANG` (`es` default, `en` on Fat Man):

```bash
ssh pirat28@100.97.87.86 'cd ~/marley1/meaning && python3 -c "import packet; print(packet.node_fingerprint())"'
```

## Human path

There isn't a useful one — each receiver is a headless Flask service under a service manager
and each sender is a CLI. Manual `node.py serve` in a terminal works but the managed unit
already owns `:8082`; stop it first or you'll get `Address already in use`.

## Gotchas (battle scars — these cost real time)

- **The mesh is symmetric, but the two receivers are managed differently.** Little Boy's
  `meaning-render` is a **system** unit (`sudo systemctl ...`). Fat Man's is a **user** unit
  (`meaning-render`, alongside `meaning-llama`, linger enabled) → prefix
  `export XDG_RUNTIME_DIR=/run/user/$(id -u)` before any `systemctl --user`. They render
  **different languages via different llama ports**: Little Boy Spanish via `:8081`, Fat Man
  English via `:8080` (Fat Man has no `:8081`). Don't "fix" the ports to match.
- **Jetson GPU is a trap for this model.** llama-server on Fat Man must run **CPU-only via
  `-dev none`**. `-ngl 0` and `CUDA_VISIBLE_DEVICES=` do **not** force CPU — the Orin's
  *integrated* GPU is always CUDA device 0, and the auto-`-fit` logic puts tensors on it
  anyway. It then dies with `NvMapMemAllocInternalTagged ... error 12` / `cudaMalloc failed`.
  `-dev none` is the only thing that worked.
- **LAN TCP between the nodes is firewalled.** Little Boy's `ufw` policy is DROP with only
  `tailscale0` allowed, so `192.168.88.x:8082` (and `:22`) **time out** even though ICMP
  ping succeeds. Use the **Tailscale** `100.x` addresses for everything. The forward leg's
  `PEER_URL` defaults to `100.110.181.128:8082`; the reverse leg overrides
  `MEANING_PEER_URL=http://100.97.87.86:8082/packet`.
- **SSH to the nodes drops on long commands.** Anything that blocks >~20s tends to return
  exit 255 mid-stream. Add `-o ServerAliveInterval=10`; the driver keeps each SSH call short.
- **Identity is reused, never regenerated.** Keys live at `~/.meaninglayer/key` and the
  pinned peer at `~/.meaninglayer/peers.json`, **outside** the repo. Fat Man's fingerprint is
  `cc5315d880d69787`; Little Boy's is `cb9246626aed27bf` (created once for the reverse leg,
  pinned on Fat Man — not regenerated since). `packet._identity()` raises if the key is
  missing rather than minting a new one — that's intentional.
- **Priority drives the prefix, not intent.** `[ALERTA]`/`[ALERT]` requires the encoder to
  classify `pri=emergency`. The storm message reliably does; calmer messages come back
  `info`/`routine`. Set `MEANING_EXPECT_PREFIX` accordingly (and to the English set on the
  reverse leg).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FAIL: sender llama :8080 down` | `ssh pirat28@100.97.87.86 'export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user restart meaning-llama'`; wait ~20s for model load. |
| `FAIL: meaning-render not active` (forward) | Little Boy system unit: `ssh marley1@100.110.181.128 'sudo systemctl restart meaning-render'`; check `journalctl -u meaning-render`. |
| `FAIL: meaning-render not active` (reverse) | Fat Man user unit: `ssh pirat28@100.97.87.86 'export XDG_RUNTIME_DIR=/run/user/$(id -u); systemctl --user restart meaning-render'`; check `journalctl --user -u meaning-render`. |
| Send hangs then `ConnectTimeout ... 192.168.88.x` | You're using a LAN IP. Use the `100.x` Tailscale address. |
| `node identity key missing: ~/.meaninglayer/key` | The signing key is gone — do **not** regenerate blindly (it's the node identity and the peer pins its pubkey). Restore from backup. |
| Receiver returns `400 verification failed` | Sender's pubkey isn't pinned on the receiver (`~/.meaninglayer/peers.json`), or the packet was tampered. Re-pin the correct `{fingerprint: pubkey}`. |
| Render service won't bind (`Address already in use`) | A manual `node.py serve` is holding `:8082`; the managed unit owns it. `fuser -k 8082/tcp` then restart the unit. |
