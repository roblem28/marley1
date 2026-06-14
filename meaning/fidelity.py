#!/usr/bin/env python3
"""Marley1 meaning/ fidelity harness -- round-trip semantic drift MEASUREMENT.

Renders paraphrase loosely (we observed "Ya voy, cinco minutos" return as
"In five minutes, it's time to leave"). This harness quantifies that drift by
sending an English message the full symmetric mesh round trip and scoring how
much meaning survives. It does NOT try to fix the drift -- the number is the
deliverable.

Round trip (real keys, real signatures verified at every hop, all local LLMs):

    Fat Man  encode+sign (en)                 [leg 1: local llama :8080, Fat Man key]
      -> Little Boy verify+render -> Spanish   [leg 2: LB llama :8081]
      -> Little Boy encode+sign (es)           [leg 3: LB llama :8081, Little Boy key]
      -> Fat Man  verify+render -> English     [leg 4: local llama :8080]

Legs 2+3 run ON Little Boy over SSH: LB's render llama (:8081, bound to
127.0.0.1) and its ed25519 identity key both live there and are unreachable
from Fat Man. This script ships its own `remote-leg` subcommand to LB (copied
to /tmp) to perform those legs in-process with the real LB identity.

Two LOCAL drift metrics (no cloud):
  * embedding cosine similarity between the source and the round-tripped English,
    using nomic-embed-text already on Fat Man. If it is not already served, this
    script loads it in llama.cpp embedding mode on a temp port and tears it down
    after the run.
  * structured-field survival: did `intent` and `pri` come back unchanged across
    the round trip? (binary per field)
  Fallback if the embedding model cannot be served: token-overlap F1 -- a weaker
  LEXICAL proxy, clearly labelled as such in the report.

Report row per message: source, spanish, roundtrip_en, cosine_sim,
intent_preserved, pri_preserved.

Usage:
    fidelity.py run             # v0.1 round trip over the test set, print JSON
    fidelity.py compare         # (alias --v2) run the test set under BOTH v0.1 and v0.2
    fidelity.py remote-leg      # (internal) v0.1 legs 2+3 on Little Boy; pkt1 JSON on stdin
    fidelity.py remote-leg-v2   # (internal) v0.2 legs 2+3 on Little Boy; pkt1 JSON on stdin

v0.2 round trip mirrors v0.1 but uses encode_v2/render_v2: the language-neutral
frame is re-rendered while verbatim spans (amounts, dates/times, proper nouns,
quantities, codes) are spliced back in literally -- so exact content survives.
"""
import os, sys, json, re, math, time, subprocess, signal

# Make `import packet` work whether we run from ~/marley1/meaning (Fat Man) or
# from /tmp (the copy shipped to Little Boy for the remote leg).
MEANING_DIR = os.path.expanduser("~/marley1/meaning")
if MEANING_DIR not in sys.path:
    sys.path.insert(0, MEANING_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- topology -----------------------------------------------------------------
LB_SSH = "marley1@100.110.181.128"
LB_LLAMA = "http://127.0.0.1:8081/v1/chat/completions"   # Little Boy render/encode llama
FATMAN_LLAMA = "http://127.0.0.1:8080/v1/chat/completions"  # Fat Man encode/render llama

# --- embedding metric config --------------------------------------------------
EMB_BIN = "/home/pirat28/llama.cpp/build/bin/llama-server"
EMB_MODEL = "/mnt/ssd/models/nomic-embed-text-v1.5.Q4_K_M.gguf"
EMB_PORT = 8090
EMB_URL = f"http://127.0.0.1:{EMB_PORT}/v1/embeddings"

DRIFT_FLOOR = 0.85  # flag rows below this similarity

# 8-message test set spanning the failure modes the protocol's research agenda names.
TEST_SET = [
    ("instruction",  "Please restock the first-aid kit before your shift ends."),
    ("emergency",    "Gas leak in the east wing -- evacuate the building now."),
    ("question",     "Has the shipment from the supplier arrived yet?"),
    ("idiom",        "Don't jump the gun on the layoffs -- wait for the board."),
    ("deadline_num", "Wire the $4,200 deposit before the 3pm cutoff today."),
    ("proper_noun",  "Tell Hudson the meeting moved to Room 204."),
    ("sarcastic",    "Oh, fantastic -- another Monday all-hands. Can't wait."),
    ("ambiguous",    "She told her she'd take care of it before they left."),
]


# ============================================================================
# Little Boy side: legs 2 + 3 (verify+render ES, then re-encode+sign ES)
# ============================================================================
def remote_leg():
    """Run on Little Boy. Reads pkt1 JSON on stdin, emits {spanish, pkt2} JSON.

    leg 2: verify Fat Man's signature (peers.json lookup) + render to Spanish.
    leg 3: re-encode that Spanish and sign it with Little Boy's identity key.
    Uses MEANING_LLAMA_URL (set by the caller to LB :8081) for both LLM calls.
    """
    import packet as mp
    pkt1 = json.load(sys.stdin)
    verify_ok = mp.verify(pkt1)                 # against pinned Fat Man pubkey
    spanish = mp.render(pkt1, "es")             # leg 2 render (LB :8081)
    pkt2 = mp.sign(mp.encode(spanish, "es"))    # leg 3 encode+sign (Little Boy key)
    json.dump({"verify_leg2": verify_ok, "spanish": spanish, "pkt2": pkt2}, sys.stdout)


def remote_leg_v2():
    """Run on Little Boy. Legs 2+3 for packet schema v0.2: verify, render_v2 to Spanish,
    re-encode_v2 that Spanish, sign with Little Boy's key. pkt1 JSON on stdin."""
    import packet as mp
    pkt1 = json.load(sys.stdin)
    verify_ok = mp.verify(pkt1)
    spanish = mp.render_v2(pkt1, "es")              # leg 2 render (two-channel)
    pkt2 = mp.sign(mp.encode_v2(spanish, "es"))     # leg 3 encode_v2 + sign (Little Boy key)
    json.dump({"verify_leg2": verify_ok, "spanish": spanish, "pkt2": pkt2}, sys.stdout)


# ============================================================================
# Fat Man side: orchestration + scoring
# ============================================================================
def _ship_to_little_boy():
    """Copy this script to Little Boy /tmp so it can run the remote leg there."""
    subprocess.run(["scp", "-q", os.path.abspath(__file__),
                    f"{LB_SSH}:/tmp/fidelity.py"], check=True, timeout=60)


def _little_boy_legs(pkt1, remote="remote-leg"):
    """Drive legs 2+3 on Little Boy via SSH; return {spanish, pkt2, verify_leg2}.

    remote: "remote-leg" (v0.1) or "remote-leg-v2" (v0.2) subcommand.
    """
    cmd = (f"cd ~/marley1/meaning && MEANING_LLAMA_URL={LB_LLAMA} "
           f"python3 /tmp/fidelity.py {remote}")
    p = subprocess.run(["ssh", LB_SSH, cmd], input=json.dumps(pkt1).encode(),
                       capture_output=True, timeout=420)
    if p.returncode != 0:
        raise RuntimeError(f"Little Boy leg failed: {p.stderr.decode()[:500]}")
    return json.loads(p.stdout.decode())


# --- embedding server lifecycle ----------------------------------------------
def _emb_alive(url):
    import requests
    try:
        r = requests.post(url, json={"input": "ping"}, timeout=5)
        return r.status_code == 200 and "data" in r.json()
    except Exception:
        return False


def ensure_embeddings():
    """Return (url, proc). proc is None if an embedding server was already up.

    Checks whether nomic-embed is already served on the temp port; if not, loads
    it in llama.cpp embedding mode. Caller must tear down a non-None proc.
    """
    if _emb_alive(EMB_URL):
        return EMB_URL, None
    if not (os.path.exists(EMB_BIN) and os.path.exists(EMB_MODEL)):
        return None, None  # signal lexical fallback
    log = open("/tmp/fidelity_embsrv.log", "w")
    proc = subprocess.Popen(
        [EMB_BIN, "-m", EMB_MODEL, "--host", "127.0.0.1", "--port", str(EMB_PORT),
         "--embeddings", "-ngl", "0", "-t", "6", "-c", "2048"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        if _emb_alive(EMB_URL):
            return EMB_URL, proc
        if proc.poll() is not None:
            break
        time.sleep(1)
    teardown_embeddings(proc)
    return None, None


def teardown_embeddings(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


# --- metrics ------------------------------------------------------------------
def _embed(text, url):
    import requests
    r = requests.post(url, json={"input": text}, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def cosine_sim(a, b, url):
    va, vb = _embed(a, url), _embed(b, url)
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    return dot / (na * nb) if na and nb else 0.0


_TOK = re.compile(r"[a-z0-9]+")


def token_overlap_f1(a, b):
    """Weaker LEXICAL proxy: set-based token-overlap F1 (used only if embeddings unavailable)."""
    sa, sb = set(_TOK.findall(a.lower())), set(_TOK.findall(b.lower()))
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(sb), inter / len(sa)
    return 2 * prec * rec / (prec + rec)


# --- one round trip -----------------------------------------------------------
def round_trip(src, emb_url):
    import packet as mp  # local: default LLAMA_URL = Fat Man :8080
    pkt1 = mp.sign(mp.encode(src, "en"))                 # leg 1
    lb = _little_boy_legs(pkt1)                           # legs 2+3
    spanish, pkt2 = lb["spanish"], lb["pkt2"]
    verify_leg4 = mp.verify(pkt2)                         # leg 4 verify (Little Boy sig)
    roundtrip_en = mp.render(pkt2, "en")                  # leg 4 render
    if emb_url:
        score = cosine_sim(src, roundtrip_en, emb_url)
    else:
        score = token_overlap_f1(src, roundtrip_en)
    return {
        "source": src,
        "spanish": spanish,
        "roundtrip_en": roundtrip_en,
        "score": score,
        "intent_src": pkt1["intent"], "intent_rt": pkt2["intent"],
        "pri_src": pkt1["pri"], "pri_rt": pkt2["pri"],
        "intent_preserved": pkt1["intent"] == pkt2["intent"],
        "pri_preserved": pkt1["pri"] == pkt2["pri"],
        "verify_leg2": lb.get("verify_leg2"), "verify_leg4": verify_leg4,
    }


def round_trip_v2(src, emb_url):
    """v0.2 round trip via encode_v2/render_v2. Same legs, two-channel packets."""
    import packet as mp
    pkt1 = mp.sign(mp.encode_v2(src, "en"))                       # leg 1
    lb = _little_boy_legs(pkt1, remote="remote-leg-v2")           # legs 2+3
    spanish, pkt2 = lb["spanish"], lb["pkt2"]
    verify_leg4 = mp.verify(pkt2)                                 # leg 4 verify
    roundtrip_en = mp.render_v2(pkt2, "en")                       # leg 4 render
    if emb_url:
        score = cosine_sim(src, roundtrip_en, emb_url)
    else:
        score = token_overlap_f1(src, roundtrip_en)
    # verbatim survival: did every exact span extracted from the source survive
    # literally into the round-tripped English output?
    src_spans = [s for vals in pkt1["verbatim"].values() for s in vals]
    spans_survived = all(s in roundtrip_en for s in src_spans)
    return {
        "source": src,
        "spanish": spanish,
        "roundtrip_en": roundtrip_en,
        "score": score,
        "intent_src": pkt1["intent"], "intent_rt": pkt2["intent"],
        "pri_src": pkt1["pri"], "pri_rt": pkt2["pri"],
        "speech_act_src": pkt1["speech_act"], "register_src": pkt1["register"],
        "intent_preserved": pkt1["intent"] == pkt2["intent"],
        "pri_preserved": pkt1["pri"] == pkt2["pri"],
        "src_spans": src_spans, "n_spans": len(src_spans),
        "spans_survived": spans_survived,
        "verify_leg2": lb.get("verify_leg2"), "verify_leg4": verify_leg4,
    }


def run():
    _ship_to_little_boy()
    emb_url, emb_proc = ensure_embeddings()
    metric = "embedding_cosine (nomic-embed-text v1.5)" if emb_url \
        else "token_overlap_f1 (LEXICAL FALLBACK -- weaker proxy)"
    rows = []
    try:
        for cat, src in TEST_SET:
            print(f"# {cat}: {src}", file=sys.stderr, flush=True)
            r = round_trip(src, emb_url)
            r["category"] = cat
            rows.append(r)
    finally:
        teardown_embeddings(emb_proc)
    print(json.dumps({"metric": metric, "rows": rows}, ensure_ascii=False))


def compare():
    """Run the SAME 8-message test set under BOTH v0.1 and v0.2, same cosine metric."""
    _ship_to_little_boy()
    emb_url, emb_proc = ensure_embeddings()
    metric = "embedding_cosine (nomic-embed-text v1.5)" if emb_url \
        else "token_overlap_f1 (LEXICAL FALLBACK -- weaker proxy)"
    rows = []
    try:
        for cat, src in TEST_SET:
            print(f"# {cat} [v1]: {src}", file=sys.stderr, flush=True)
            v1 = round_trip(src, emb_url)
            print(f"# {cat} [v2]: {src}", file=sys.stderr, flush=True)
            v2 = round_trip_v2(src, emb_url)
            rows.append({"category": cat, "v1": v1, "v2": v2})
    finally:
        teardown_embeddings(emb_proc)
    print(json.dumps({"metric": metric, "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "remote-leg":
        remote_leg()
    elif mode == "remote-leg-v2":
        remote_leg_v2()
    elif mode == "run":
        run()
    elif mode in ("compare", "--v2"):
        compare()
    else:
        print("usage: fidelity.py run | compare | remote-leg | remote-leg-v2", file=sys.stderr)
        sys.exit(1)
