#!/usr/bin/env python3
"""Marley1 meaning/ node runner -- the thin transport around packet.py.

Usage:
    node.py send "english text" [src_lang]   # sender: encode + sign + POST to peer
    node.py serve                            # receiver: Flask verify + render

Decoupled from marley_server.py by design: this is a standalone message type.

Receiver env (serve):
    MEANING_RENDER_LANG  target render language + prefix set (default "es";
                         "es" -> [ALERTA]/[INFO]/[RUTINA], "en" -> [ALERT]/[INFO]/[ROUTINE])
    MEANING_PORT         listen port (default 8082)
    MEANING_LLAMA_URL    llama-server used for rendering (Little Boy :8081, Fat Man :8080)
    MEANING_INBOX        append-log of accepted renders
"""
import sys, os, json, datetime, requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packet as mp  # noqa: E402

INBOX = os.path.expanduser(os.environ.get("MEANING_INBOX", "~/meaninglayer/inbox.log"))


def send(text, src_lang="en"):
    pkt = mp.sign(mp.encode(text, src_lang))
    print(f"node identity (from): {pkt['from']}")
    s = mp.sizes(text, pkt)
    print("--- byte sizes ---")
    print(f"raw text     : {s['raw_text']}")
    print(f"gzip(text)   : {s['gzip_text']}")
    print(f"packet JSON  : {s['packet_json']}")
    print(f"gzip(packet) : {s['gzip_packet']}")
    print(f"intent={pkt['intent']} pri={pkt['pri']}")
    print(f"POST {mp.PEER_URL}")
    r = requests.post(mp.PEER_URL, data=json.dumps(pkt, separators=(",", ":")),
                      headers={"Content-Type": "application/json"}, timeout=180)
    print(f"status {r.status_code}")
    print(r.text)


def serve():
    from flask import Flask, request, jsonify
    lang = os.environ.get("MEANING_RENDER_LANG", "es")   # es=Little Boy, en=Fat Man reverse leg
    port = int(os.environ.get("MEANING_PORT", "8082"))
    app = Flask(__name__)

    @app.post("/packet")
    def _packet():
        pkt = request.get_json(force=True)
        # verify ed25519 signature against the sender pubkey pinned in peers.json
        if not mp.verify(pkt, mp._peers().get(pkt.get("from", ""))):
            print(f"REJECTED {pkt.get('id')}: bad or unknown signature", flush=True)
            return jsonify({"ok": False, "error": "verification failed"}), 400
        out = mp.render(pkt, lang)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(f"\n{out}\n", flush=True)
        with open(INBOX, "a") as f:
            f.write(f"{ts}\t{pkt['id']}\t{out}\n")
        # "render"/"lang" are the canonical keys; "es" kept for backward-compat with old clients.
        return jsonify({"ok": True, "lang": lang, "render": out, "es": out})

    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        serve()
    elif len(sys.argv) >= 3 and sys.argv[1] == "send":
        send(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "en")
    else:
        print('usage: node.py send "text" [src_lang]  |  node.py serve')
        sys.exit(1)
