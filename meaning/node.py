#!/usr/bin/env python3
"""Marley1 meaning/ node runner -- the thin transport around packet.py.

Usage:
    node.py send "english text" [src_lang]   # sender: encode + sign + POST to peer
    node.py serve                            # receiver: Flask :8082 verify + render

Decoupled from marley_server.py by design: this is a standalone message type.
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
    app = Flask(__name__)

    @app.post("/packet")
    def _packet():
        pkt = request.get_json(force=True)
        # verify ed25519 signature against the sender pubkey pinned in peers.json
        if not mp.verify(pkt, mp._peers().get(pkt.get("from", ""))):
            print(f"REJECTED {pkt.get('id')}: bad or unknown signature", flush=True)
            return jsonify({"ok": False, "error": "verification failed"}), 400
        es = mp.render(pkt, "es")
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(f"\n{es}\n", flush=True)
        with open(INBOX, "a") as f:
            f.write(f"{ts}\t{pkt['id']}\t{es}\n")
        return jsonify({"ok": True, "es": es})

    app.run(host="0.0.0.0", port=8082)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        serve()
    elif len(sys.argv) >= 3 and sys.argv[1] == "send":
        send(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "en")
    else:
        print('usage: node.py send "text" [src_lang]  |  node.py serve')
        sys.exit(1)
