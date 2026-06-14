#!/usr/bin/env python3
"""Marley1 meaning/ -- MTP v0.1 "meaning packet" message type + ed25519 node identity.

Public API:
    encode(text, src_lang="en") -> dict      # LLM semantic extraction -> unsigned packet
    sign(packet) -> dict                      # attach ed25519 signature (node identity)
    verify(packet, peer_pubkey) -> bool       # verify sig against pinned peer pubkey
    render(packet, target_lang="es") -> str   # natural-language render, priority-prefixed

Node identity (the trust primitive) reuses the existing demo material -- never regenerated:
    key   : ~/.meaninglayer/key          (hex-encoded 32-byte ed25519 seed; never committed)
    peers : ~/.meaninglayer/peers.json   ({fingerprint: pubkey_hex})
"""
import os, json, uuid, gzip, hashlib, datetime, importlib.util, requests
from nacl import signing, encoding
from nacl.exceptions import BadSignatureError

# --- config (env-overridable) -------------------------------------------------
LLAMA_URL = os.environ.get("MEANING_LLAMA_URL", "http://127.0.0.1:8080/v1/chat/completions")
PEER_URL = os.environ.get("MEANING_PEER_URL", "http://100.110.181.128:8082/packet")
KEYFILE = os.path.expanduser(os.environ.get("MEANING_KEY", "~/.meaninglayer/key"))
PEERSFILE = os.path.expanduser(os.environ.get("MEANING_PEERS", "~/.meaninglayer/peers.json"))

# task 3 seam: route sem.summary through ~/marley1/compression/abbrev.py when True.
# Leave OFF -- this is only the wiring, not the optimization.
USE_ABBREV = False
ABBREV_PATH = os.path.expanduser("~/marley1/compression/abbrev.py")

MP_VERSION = "0.1"
# Priority prefixes per render language. Spanish is the original v0.1 set; English is the
# symmetric reverse-leg equivalent (emergency->ALERT, info->INFO, routine->ROUTINE).
PREFIXES = {
    "es": {"emergency": "[ALERTA]", "info": "[INFO]", "routine": "[RUTINA]"},
    "en": {"emergency": "[ALERT]", "info": "[INFO]", "routine": "[ROUTINE]"},
}
PREFIX = PREFIXES["es"]  # back-compat: bare PREFIX keeps the original Spanish set
LANG_NAME = {"es": "espanol", "en": "english", "fr": "francais"}

# Render system prompt per target language. The Spanish text is unchanged from v0.1;
# English is added so a node can render the meaning core back to English (reverse leg).
RENDER_SYS = {
    "es": ("Eres un traductor semantico. A partir del nucleo semantico dado, "
           "escribe UN parrafo corto en {lang} natural que preserve la intencion "
           "y la prioridad del mensaje. Devuelve solo el parrafo, sin notas ni etiquetas."),
    "en": ("You are a semantic translator. From the given semantic core, write ONE "
           "short paragraph in natural {lang} that preserves the intent and priority "
           "of the message. Return only the paragraph, with no notes or labels."),
}

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "entities": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "intent": {"type": "string",
                   "enum": ["alert", "question", "instruction", "observation", "discussion"]},
        "pri": {"type": "string", "enum": ["emergency", "info", "routine"]},
    },
    "required": ["summary", "entities", "actions", "intent", "pri"],
}

SYS_EXTRACT = ("You extract the semantic core of a message. Return JSON only. "
               "summary: terse English gist of the meaning. "
               "entities: key nouns/places/objects mentioned. "
               "actions: required actions as short imperatives. "
               "intent: alert|question|instruction|observation|discussion. "
               "pri: emergency = an urgent threat to safety or property, or a hard "
               "deadline demanding immediate action (storms, hazards, evacuations, "
               "'by 3pm', protect/secure equipment); info = noteworthy update needing "
               "no immediate action; routine = mundane, low-stakes. "
               "When a message warns of danger AND requires timely protective action, "
               "use emergency.")


# === packet v0.2 : two-channel (verbatim + renderable frame) ==================
# v0.2 splits a message into two channels so exact content survives a round trip
# even though the frame is freely re-rendered into another language:
#   verbatim{}        -- exact spans copied char-for-char, NEVER translated/reworded.
#   renderable.summary-- the language-neutral frame, with typed {bucket[i]}
#                        placeholders pointing at verbatim spans.
# render_v2 translates ONLY the frame, leaving placeholders intact, then splices
# the verbatim values back in by string replacement -- so the model never sees,
# translates, or paraphrases a number, name, or amount.
MP_VERSION_V2 = "0.2"
VERBATIM_BUCKETS = ["amounts", "dates_times", "proper_nouns", "quantities", "codes"]
SPEECH_ACTS = ["command", "request", "query", "statement", "warning"]
REGISTERS = ["neutral", "urgent", "sarcastic", "formal", "casual"]

# A tone marker appended (not "rendered") when register==sarcastic: irony does not
# survive paraphrase, so we flag it explicitly in the target language instead.
SARCASM_MARK = {"es": "(tono: sarcastico)", "en": "(tone: sarcastic)"}

SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "intent": {"type": "string",
                   "enum": ["alert", "question", "instruction", "observation", "discussion"]},
        "speech_act": {"type": "string", "enum": SPEECH_ACTS},
        "pri": {"type": "string", "enum": ["emergency", "info", "routine"]},
        "register": {"type": "string", "enum": REGISTERS},
        "verbatim": {
            "type": "object",
            "properties": {b: {"type": "array", "items": {"type": "string"}}
                           for b in VERBATIM_BUCKETS},
            "required": VERBATIM_BUCKETS,
        },
        "summary": {"type": "string"},
    },
    "required": ["intent", "speech_act", "pri", "register", "verbatim", "summary"],
}

SYS_EXTRACT_V2 = (
    "You convert a message into a two-channel semantic packet. Return JSON only.\n"
    "intent: alert|question|instruction|observation|discussion.\n"
    "speech_act: command|request|query|statement|warning -- the actual act being "
    "performed (a yes/no question is query; an order is command; a polite ask is request; "
    "a hazard notice is warning).\n"
    "pri: emergency = urgent threat to safety/property OR a hard deadline demanding "
    "immediate action; info = noteworthy, no immediate action; routine = mundane.\n"
    "register: neutral|urgent|sarcastic|formal|casual. Use sarcastic when the literal "
    "words are positive/enthusiastic but the real attitude is negative or ironic -- e.g. "
    "praise words ('great', 'fantastic', 'wonderful', 'perfect', 'can't wait') attached to "
    "an unwanted, tedious, or bad event signal sarcasm, not genuine enthusiasm.\n"
    "verbatim: EXACT substrings copied character-for-character FROM THE MESSAGE, never "
    "reworded, translated, normalized, or invented. Buckets:\n"
    "  amounts: money/prices, e.g. \"$4,200\".\n"
    "  dates_times: dates, times, deadlines, e.g. \"3pm\", \"today\", \"Monday\".\n"
    "  proper_nouns: names of people/places/orgs/rooms, e.g. \"Hudson\", \"Room 204\".\n"
    "  quantities: numeric quantities/measures, e.g. \"5\", \"two boxes\".\n"
    "  codes: identifiers/SKUs/reference codes.\n"
    "  Use [] for an empty bucket. Copy spans EXACTLY as written.\n"
    "summary: a faithful paraphrase of the FULL message that preserves the speech act, "
    "but with EVERY verbatim span replaced by a typed placeholder of the form "
    "{bucket[index]} -- e.g. {amounts[0]}, {dates_times[0]}, {proper_nouns[1]}. "
    "RULES for placeholders: (1) every placeholder MUST have a numeric index in square "
    "brackets, like {dates_times[0]} -- never a bare {dates_times}; (2) use ONLY these "
    "bucket names: amounts, dates_times, proper_nouns, quantities, codes -- NEVER invent a "
    "placeholder for any other field (no {speech_act}, {intent}, etc.); (3) write all other "
    "meaning as ordinary words, not placeholders. "
    "The summary MUST contain one placeholder for every span you listed in verbatim, and "
    "MUST NOT contain the literal span text. Keep a question phrased as a question, a "
    "command as a command.")

# Render the v0.2 frame only. The model translates surrounding prose but must echo every
# {bucket[i]} placeholder byte-for-byte (verbatim values are spliced in afterwards).
RENDER_SYS_V2 = {
    "es": ("Eres un traductor. Traduce el texto a espanol natural y conciso. El texto "
           "contiene marcadores entre llaves como {amounts[0]} o {proper_nouns[0]}: "
           "REPRODUCELOS EXACTAMENTE IGUAL, sin traducirlos, sin modificarlos y sin "
           "quitar las llaves. No anadas informacion. Devuelve solo el texto traducido."),
    "en": ("You are a translator. Translate the text into natural, concise english. The "
           "text contains placeholders in braces like {amounts[0]} or {proper_nouns[0]}: "
           "REPRODUCE THEM EXACTLY, do not translate, alter, or drop the braces. Add no "
           "information. Return only the translated text."),
}


# --- identity / peers ---------------------------------------------------------
def _identity():
    """Load (never create) the ed25519 node identity signing key."""
    if not os.path.exists(KEYFILE):
        raise FileNotFoundError(f"node identity key missing: {KEYFILE}")
    seed = open(KEYFILE, "rb").read().strip()
    return signing.SigningKey(seed, encoder=encoding.HexEncoder)


def fingerprint(pubkey_bytes):
    return hashlib.sha256(pubkey_bytes).hexdigest()[:16]


def node_fingerprint():
    return fingerprint(_identity().verify_key.encode())


def _peers():
    return json.load(open(PEERSFILE)) if os.path.exists(PEERSFILE) else {}


# --- LLM helper ---------------------------------------------------------------
def _llm(messages, temperature=0, max_tokens=512, response_format=None):
    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if response_format:
        body["response_format"] = response_format
    r = requests.post(LLAMA_URL, json=body, timeout=180)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _maybe_abbrev(summary):
    """Task 3 seam: compress sem.summary via abbrev.py if enabled & importable, else LLM summary."""
    if not USE_ABBREV:
        return summary
    try:
        spec = importlib.util.spec_from_file_location("abbrev", ABBREV_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ConstructionCompressor().compress(summary)
    except Exception:
        return summary  # fall back to the LLM summary on any failure


# --- canonical signing payload ------------------------------------------------
def canon(intent, pri, sem):
    """Canonical bytes the sender signs and the receiver verifies (v0.1)."""
    return json.dumps({"intent": intent, "pri": pri, "sem": sem},
                      sort_keys=True, separators=(",", ":")).encode()


def canon_v2(packet):
    """Canonical bytes for v0.2 -- covers the WHOLE semantic payload (both channels)."""
    return json.dumps({
        "intent": packet["intent"],
        "speech_act": packet["speech_act"],
        "pri": packet["pri"],
        "register": packet["register"],
        "src_lang": packet.get("src_lang", ""),
        "verbatim": packet["verbatim"],
        "renderable": packet["renderable"],
    }, sort_keys=True, separators=(",", ":")).encode()


def _canon_payload(packet):
    """Version-gated signing payload. v0.1 is byte-identical to the original canon()."""
    if packet.get("mp") == MP_VERSION_V2:
        return canon_v2(packet)
    return canon(packet["intent"], packet["pri"], packet["sem"])


# --- public API ---------------------------------------------------------------
def encode(text, src_lang="en"):
    """English text -> unsigned MTP v0.1 packet (semantic core via local LLM)."""
    e = json.loads(_llm(
        [{"role": "system", "content": SYS_EXTRACT}, {"role": "user", "content": text}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "sem", "strict": True, "schema": SCHEMA}}))
    sem = {"summary": _maybe_abbrev(e["summary"]),
           "entities": e["entities"], "actions": e["actions"]}
    return {
        "mp": MP_VERSION,
        "id": str(uuid.uuid4()),
        "from": node_fingerprint(),
        "sig": "",
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "intent": e["intent"],
        "pri": e["pri"],
        "src_lang": src_lang,
        "sem": sem,
    }


def sign(packet):
    """Attach an ed25519 signature using this node's identity key (v0.1 + v0.2)."""
    sk = _identity()
    packet["sig"] = sk.sign(_canon_payload(packet)).signature.hex()
    return packet


def verify(packet, peer_pubkey=None):
    """Verify the packet signature. peer_pubkey: hex string, or None to look up via peers.json."""
    pub_hex = peer_pubkey or _peers().get(packet.get("from", ""))
    if not pub_hex:
        return False
    try:
        vk = signing.VerifyKey(pub_hex, encoder=encoding.HexEncoder)
        vk.verify(_canon_payload(packet), bytes.fromhex(packet["sig"]))
        return True
    except (BadSignatureError, ValueError, KeyError):
        return False


def render(packet, target_lang="es"):
    """Render packet.sem into one short natural-language paragraph, priority-prefixed.

    target_lang selects both the prose language and the priority-prefix set:
    es -> [ALERTA]/[INFO]/[RUTINA], en -> [ALERT]/[INFO]/[ROUTINE].
    """
    sem = packet["sem"]
    lang = LANG_NAME.get(target_lang, target_lang)
    sysmsg = RENDER_SYS.get(target_lang, RENDER_SYS["es"]).format(lang=lang)
    user = (f"intent={packet['intent']} pri={packet['pri']}\n"
            f"summary: {sem['summary']}\n"
            f"entities: {', '.join(sem.get('entities', []))}\n"
            f"actions: {', '.join(sem.get('actions', []))}")
    out = _llm([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}],
               temperature=0.3, max_tokens=256).strip()
    pfx = PREFIXES.get(target_lang, PREFIX)
    return f"{pfx.get(packet['pri'], '[INFO]')} {out}"


# --- v0.2 public API ----------------------------------------------------------
import re as _re  # noqa: E402  (local alias; keeps the v0.1 import block untouched)
_PLACEHOLDER = _re.compile(r"\{(\w+)\[(\d+)\]\}")


def _substitute(text, verbatim):
    """Splice verbatim spans back into a rendered frame by literal string replacement.

    {bucket[i]} -> verbatim[bucket][i], char-for-char. Any leftover placeholder the
    model emitted malformed (bare {quantities}, or an invented {speech_act}) or that is
    out of range is stripped -- never leaked as a raw token -- and whitespace is tidied.
    """
    def repl(m):
        bucket, idx = m.group(1), int(m.group(2))
        vals = verbatim.get(bucket, [])
        return vals[idx] if idx < len(vals) else ""
    text = _PLACEHOLDER.sub(repl, text)              # resolve well-formed {bucket[i]}
    text = _re.sub(r"\{[^{}]*\}", "", text)          # drop any malformed leftover braces
    return _re.sub(r"\s{2,}", " ", text).strip()     # tidy gaps left by removed tokens


def encode_v2(text, src_lang="en"):
    """Text -> unsigned MTP v0.2 packet: two-channel (verbatim + renderable frame).

    One JSON-constrained llama call extracts intent/speech_act/pri/register, the exact
    verbatim spans, and a language-neutral summary whose every verbatim span is a typed
    {bucket[i]} placeholder. Sign with sign() exactly as v0.1.
    """
    e = json.loads(_llm(
        [{"role": "system", "content": SYS_EXTRACT_V2}, {"role": "user", "content": text}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "sem2", "strict": True, "schema": SCHEMA_V2}}))
    verbatim = {b: list(e["verbatim"].get(b, [])) for b in VERBATIM_BUCKETS}
    return {
        "mp": MP_VERSION_V2,
        "id": str(uuid.uuid4()),
        "from": node_fingerprint(),
        "sig": "",
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "intent": e["intent"],
        "speech_act": e["speech_act"],
        "pri": e["pri"],
        "register": e["register"],
        "src_lang": src_lang,
        "verbatim": verbatim,
        "renderable": {"summary": e["summary"]},
    }


def render_v2(packet, target_lang="es"):
    """Render a v0.2 packet: translate ONLY the frame, then splice verbatim spans back.

    The model never sees the verbatim values (numbers, names, amounts) -- it only moves
    placeholders around -- so they cannot be paraphrased or mistranslated. A sarcastic
    register is flagged with an explicit tone marker rather than rendered as irony.
    """
    summary = packet["renderable"]["summary"]
    sysmsg = RENDER_SYS_V2.get(target_lang, RENDER_SYS_V2["es"])
    framed = _llm([{"role": "system", "content": sysmsg}, {"role": "user", "content": summary}],
                  temperature=0.2, max_tokens=256).strip()
    out = _substitute(framed, packet["verbatim"]).strip()
    pfx = PREFIXES.get(target_lang, PREFIX).get(packet["pri"], "[INFO]")
    if packet.get("register") == "sarcastic":
        out = f"{out} {SARCASM_MARK.get(target_lang, SARCASM_MARK['en'])}"
    return f"{pfx} {out}"


def sizes(text, packet):
    """Byte accounting: raw text, gzip(text), packet JSON, gzip(packet)."""
    raw = text.encode()
    pj = json.dumps(packet, separators=(",", ":")).encode()
    return {"raw_text": len(raw), "gzip_text": len(gzip.compress(raw)),
            "packet_json": len(pj), "gzip_packet": len(gzip.compress(pj))}
