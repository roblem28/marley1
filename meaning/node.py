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

Send env:
    SEND_V2=1            opt into packet schema v0.2 (two-channel verbatim/renderable);
                         default unset -> v0.1, so existing senders are unchanged.

Respond env (serve):
    RESPOND_MODE=1       the node ANSWERS as Marley: on an inbound packet it understands
                         the message, generates a reply with its local LLM, and sends that
                         reply BACK to the original sender (rendered in the sender's
                         language). Default off -> the node only re-renders, as before.

The receiver dispatches on packet.mp ("0.1" -> render, "0.2" -> render_v2), so a
v0.2-aware node and a v0.1-only node interoperate over the same :8082 endpoint.

Loopback guard: a reply packet carries "in_reply_to" = the original packet id. Respond
mode ignores any packet that already carries it, so replies never trigger replies.
"""
import sys, os, json, datetime, threading, requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packet as mp  # noqa: E402

INBOX = os.path.expanduser(os.environ.get("MEANING_INBOX", "~/meaninglayer/inbox.log"))


def send(text, src_lang="en"):
    # SEND_V2=1 opts into v0.2; default path is byte-identical to before.
    if os.environ.get("SEND_V2") == "1":
        pkt = mp.sign(mp.encode_v2(text, src_lang))
    else:
        pkt = mp.sign(mp.encode(text, src_lang))
    print(f"node identity (from): {pkt['from']}  mp={pkt['mp']}")
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


def _respond(pkt, lang):
    """Marley answers: understand the inbound message, generate a reply, sign it, and send
    it BACK to the original sender's receiver (rendered there in the sender's language).

    Runs in a background thread so the inbound POST is ack'd immediately. The reply packet
    is tagged in_reply_to so it can never itself trigger a reply (loopback guard)."""
    sender = pkt.get("from", "")
    try:
        addr = mp.reply_addr(sender)
        if not addr:
            print(f"RESPOND: no return address for {sender}; dropping reply", flush=True)
            return
        message = mp.render_for_node(pkt, lang)                 # inbound -> node's language, untagged
        reply_text = mp.marley_reply(message, sender=sender, lang=lang)
        reply = mp.sign(mp.encode(reply_text, src_lang=lang))   # NEW signed packet (reuse encode/sign)
        reply["in_reply_to"] = pkt["id"]                        # loopback tag (outside canon; sig intact)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(f"\nMARLEY REPLY -> {sender} @ {addr}\n  {reply_text}\n", flush=True)
        with open(INBOX, "a") as f:
            f.write(f"{ts}\t{reply['id']}\tREPLY(in_reply_to={pkt['id']})\t{reply_text}\n")
        r = requests.post(addr, data=json.dumps(reply, separators=(",", ":")),
                          headers={"Content-Type": "application/json"}, timeout=180)
        print(f"RESPOND: delivered reply, sender receiver status {r.status_code}", flush=True)
    except Exception as e:
        print(f"RESPOND: failed for {sender}: {e}", flush=True)


def serve():
    from flask import Flask, request, jsonify
    lang = os.environ.get("MEANING_RENDER_LANG", "es")   # es=Little Boy, en=Fat Man reverse leg
    port = int(os.environ.get("MEANING_PORT", "8082"))
    respond_mode = os.environ.get("RESPOND_MODE") == "1"
    app = Flask(__name__)
    print(f"meaning/ receiver up: lang={lang} respond_mode={respond_mode}", flush=True)

    @app.post("/packet")
    def _packet():
        pkt = request.get_json(force=True)
        # verify ed25519 signature against the sender pubkey pinned in peers.json
        if not mp.verify(pkt, mp._peers().get(pkt.get("from", ""))):
            print(f"REJECTED {pkt.get('id')}: bad or unknown signature", flush=True)
            return jsonify({"ok": False, "error": "verification failed"}), 400
        # version-gate on packet.mp: v0.1 stays exactly as before, v0.2 uses the two-channel render
        out = mp.render_v2(pkt, lang) if pkt.get("mp") == mp.MP_VERSION_V2 else mp.render(pkt, lang)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print(f"\n{out}\n", flush=True)
        with open(INBOX, "a") as f:
            f.write(f"{ts}\t{pkt['id']}\t{out}\n")
        # respond mode: answer as Marley, but NEVER reply to a reply (loopback guard)
        is_reply = "in_reply_to" in pkt
        responding = respond_mode and not is_reply
        if respond_mode and is_reply:
            print(f"loop-guard: {pkt.get('id')} is a reply (in_reply_to={pkt['in_reply_to']}); "
                  f"not answering", flush=True)
        if responding:
            threading.Thread(target=_respond, args=(pkt, lang), daemon=True).start()
        # "render"/"lang" are the canonical keys; "es" kept for backward-compat with old clients.
        return jsonify({"ok": True, "lang": lang, "render": out, "es": out,
                        "responding": responding, "is_reply": is_reply})

    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        serve()
    elif len(sys.argv) >= 3 and sys.argv[1] == "send":
        send(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "en")
    else:
        print('usage: node.py send "text" [src_lang]  |  node.py serve')
        sys.exit(1)
