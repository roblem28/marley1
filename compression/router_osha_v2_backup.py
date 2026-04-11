#!/usr/bin/env python3
"""Marley1 Neuro-Symbolic Query Router v4.1.

v4.1 changes:
- Hard answer_mode overrides for governing_reference and numeric_lookup
- Domain lexicon query expansion
- Output containment: mode-specific max_tokens
- governing_reference queries cap expansion contribution
"""

import os
import re
import time
import hashlib
import json

_GROUNDING = (
    " Answer ONLY from the provided text."
    " Each point must cite a specific section, table, or standard from the text."
    " Do not introduce numbers, standards, or distances not present in the text."
    " If the text does not specify, say not specified in provided sections."
)

DOC_SIGNATURES = {
    "electrical": {
        "patterns": [r"UFC\s+3-501", r"ELECTRICAL\s+ENGINEERING", r"electrical\s+design", r"grounding", r"switchgear", r"NFPA\s+70", r"arc\s+flash", r"transformer", r"circuit"],
        "codebook": "construction",
        "system_prompt": "You are a DoD electrical systems expert. Be specific and cite section numbers when available." + _GROUNDING,
    },
    "structural": {
        "patterns": [r"UFC\s+3-301", r"STRUCTURAL\s+ENGINEERING", r"seismic", r"wind\s+load", r"ASCE\s+7", r"progressive\s+collapse", r"foundation"],
        "codebook": "construction",
        "system_prompt": "You are a DoD structural engineering expert. Be specific about load requirements and design criteria." + _GROUNDING,
    },
    "mechanical": {
        "patterns": [r"HVAC", r"rooftop\s+unit", r"air\s+handling", r"ductwork",
                     r"commissioning", r"VAV", r"chiller", r"boiler"],
        "codebook": "construction",
        "system_prompt": "You are a DoD mechanical/HVAC systems expert. Include equipment specs and performance criteria." + _GROUNDING,
    },
    "plumbing": {
        "patterns": [r"plumbing", r"domestic\s+water", r"sanitary", r"backflow",
                     r"water\s+heater", r"drainage", r"GPM"],
        "codebook": "construction",
        "system_prompt": "You are a DoD plumbing systems expert. Be specific about sizing and code requirements." + _GROUNDING,
    },
    "fire_protection": {
        "patterns": [r"fire\s+suppression", r"sprinkler", r"NFPA\s+13", r"fire\s+alarm",
                     r"pre-action", r"standpipe", r"fire\s+pump"],
        "codebook": "construction",
        "system_prompt": "You are a DoD fire protection expert. Reference NFPA standards when applicable." + _GROUNDING,
    },
    "c5isr": {
        "patterns": [r"UFC\s+4-141", r"C5ISR", r"SURVEILLANCE", r"RECONNAISSANCE", r"COMMAND.*CONTROL", r"antenna", r"telecommunications"],
        "codebook": "military",
        "system_prompt": "You are a DoD C5ISR facilities expert. Be specific about infrastructure and security requirements." + _GROUNDING,
    },
    "housing": {
        "patterns": [r"unaccompanied\s+housing", r"barracks", r"living\s+area",
                     r"bedroom", r"common\s+area", r"laundry", r"ADA\s+compliance"],
        "codebook": "construction",
        "system_prompt": "You are a DoD housing design expert. Be specific about space requirements and amenity standards." + _GROUNDING,
    },
    "microgrid": {
        "patterns": [r"microgrid", r"islanding", r"distributed\s+energy",
                     r"energy\s+storage", r"grid\s+tie", r"renewable"],
        "codebook": "construction",
        "system_prompt": "You are a DoD microgrid and power systems expert. Be specific about control requirements and redundancy." + _GROUNDING,
    },
    "medical": {
        "patterns": [r"patient", r"diagnosis", r"treatment", r"clinical",
                     r"dosage", r"prognosis", r"ICD-10"],
        "codebook": "medical",
        "system_prompt": "You are a medical document analysis expert. Be precise with clinical terminology." + _GROUNDING,
    },
    "legal": {
        "patterns": [r"plaintiff", r"defendant", r"jurisdiction", r"statute",
                     r"tort", r"liability", r"contract\s+law"],
        "codebook": "legal",
        "system_prompt": "You are a legal document analysis expert. Reference specific clauses and sections." + _GROUNDING,
    },
    "financial": {
        "patterns": [r"revenue", r"EBITDA", r"balance\s+sheet", r"SEC\s+filing",
                     r"quarterly\s+report", r"P/?E\s+ratio", r"fiscal\s+year"],
        "codebook": "financial",
        "system_prompt": "You are a financial document analysis expert. Be precise with figures and reporting periods." + _GROUNDING,
    },
    "osha_construction": {
        "patterns": [r"OSHA", r"29\s+CFR\s+1926", r"Subpart\s+[A-Z]", r"competent\s+person",
                     r"scaffold", r"excavation", r"trench",
                     r"fall\s+protection", r"guardrail",
                     r"hard\s+hat", r"crane", r"derrick",
                     r"confined\s+space", r"protective\s+system"],
        "codebook": "construction",
        "system_prompt": "You are an OSHA construction safety expert. Reference specific 29 CFR 1926 sections. Be precise about thresholds and conditions." + _GROUNDING,
    },
}

# ---------------------------------------------------------------------------
# Answer-mode classification v4.1 -- hard overrides first
# ---------------------------------------------------------------------------
_GOVERNING_HARD = [
    r"what\s+(?:standard|code|UFC|reference|specification)\s+(?:governs|applies|covers)",
    r"which\s+(?:UFC|standard|code|NFPA|ASCE|ACI|IEEE)\b",
    r"(?:governed|covered|addressed)\s+by\s+(?:which|what)",
    r"what\s+(?:is\s+the\s+)?governing\s+(?:standard|code|reference)",
    r"per\s+which\s+(?:standard|code|UFC)",
    r"progressive\s+collapse",
    r"blast\s+resist",
    r"what\s+reference\s+applies",
    r"what\s+code\s+applies",
    r"what\s+governs",
]

_NUMERIC_HARD = [
    r"how\s+(?:much|many|large|big|tall|wide|long|thick|deep|far)",
    r"what\s+(?:is\s+the\s+)?(?:size|area|capacity|rating|voltage|distance|depth|height|width|length|temperature)\b",
    r"(?:minimum|maximum)\s+(?:size|area|space|distance|depth|height|width|length)\b",
    r"\b(?:NSF|square\s+feet|square\s+foot|kW|kVA|GPM|CFM|BTU|psi)\b",
    r"\b(?:washer|dryer|closet|laundry|storage)\b.*(?:requirement|standard|size|space|area)",
    r"(?:closet|storage|laundry)\s+(?:space|facility|room)\s+requirement",
]

_DEFINITION_PATTERNS = [
    r"what\s+(?:is\s+(?:a|an|the)\s+)?(?:definition|meaning)\s+of",
    r"define\s+\w+",
    r"what\s+does\s+.+\s+mean\b",
]

_PROCESS_PATTERNS = [
    r"how\s+(?:to|should|do\s+you|does\s+one)\s+",
    r"what\s+(?:is\s+the\s+)?(?:process|procedure|sequence|steps?|workflow)\s+for",
]

_MULTI_ASPECT_PATTERNS = [
    r"compare",
    r"difference\s+between",
    r"relationship\s+between",
    r"trade-?off",
]


def classify_answer_mode(query):
    """Classify query into answer mode. Hard overrides checked first."""
    q = query.lower().strip()

    # Hard overrides -- these fire before anything else
    for p in _GOVERNING_HARD:
        if re.search(p, q):
            return "governing_reference"

    for p in _NUMERIC_HARD:
        if re.search(p, q):
            return "numeric_lookup"

    for p in _DEFINITION_PATTERNS:
        if re.search(p, q):
            return "extractive_definition"

    for p in _PROCESS_PATTERNS:
        if re.search(p, q):
            return "process_workflow"

    for p in _MULTI_ASPECT_PATTERNS:
        if re.search(p, q):
            return "multi_aspect"

    return "requirement_list"


# ---------------------------------------------------------------------------
# Domain lexicon for query expansion
# ---------------------------------------------------------------------------
DOMAIN_LEXICON = {
    "rf":          ["radio frequency", "EMC", "isolation", "separation", "electromagnetic"],
    "grounding":   ["bonding", "equipotential", "earth electrode", "ground ring", "ground rod"],
    "bonding":     ["grounding", "equipotential", "earth electrode"],
    "metering":    ["monitoring", "CT", "PT", "SCADA", "energy management"],
    "monitoring":  ["metering", "CT", "PT", "SCADA", "energy management"],
    "islanding":   ["black start", "off-grid", "PCC", "soft transition", "grid disconnect"],
    "closet":      ["storage", "NSF", "room matrix", "wardrobe"],
    "storage":     ["closet", "NSF", "room matrix", "wardrobe"],
    "antenna":     ["RF", "radio frequency", "EMC", "isolation", "separation"],
    "cybersecurity": ["risk management framework", "RMF", "UFC 4-010-06", "NIST"],
    "progressive collapse": ["redundancy", "alternate load path", "tie force", "UFC 4-023-03"],
    "blast":       ["explosion", "protective design", "UFC 4-010-01", "antiterrorism"],
    "renewable":   ["solar", "photovoltaic", "PV", "wind", "IEEE 1547"],
    "laundry":     ["washer", "dryer", "laundry room", "laundry facility"],
    "hvac":        ["air handling", "cooling", "heating", "ventilation", "mechanical"],
    "power":       ["electrical", "distribution", "switchgear", "transformer", "UPS"],
    # OSHA / CFR terms
    "scaffold": ["platform", "guardrail", "working level", "1926.451"],
    "excavation": ["trench", "cave-in", "shoring", "sloping", "1926.651"],
    "fall protection": ["guardrail", "safety net", "personal fall arrest", "6 feet", "1926.501"],
    "ladder": ["stairway", "portable ladder", "fixed ladder", "1926.1053"],
    "crane": ["derrick", "hoist", "rigging", "1926.1400"],
    "hard hat": ["head protection", "impact", "falling object", "1926.100"],
    "confined space": ["permit", "hazardous atmosphere", "entrant", "1926.1204"],
    "fire protection": ["fire extinguisher", "alarm", "flammable", "1926.150"],
    "respiratory": ["respirator", "air purifying", "1910.134", "1926.103"],
    "safety net": ["ANSI A10.11", "fall arrest", "1926.502"],
    "competent person": ["qualified person", "inspection", "daily"],

}


def expand_query(query):
    """Expand query with domain synonyms."""
    q_lower = query.lower()
    expansions = set()
    for term, synonyms in DOMAIN_LEXICON.items():
        if re.search(r'\b' + re.escape(term) + r'\b', q_lower):
            for syn in synonyms:
                if syn.lower() not in q_lower:
                    expansions.add(syn)
    if not expansions:
        return query
    return query + " " + " ".join(sorted(expansions))


# ---------------------------------------------------------------------------
# Output containment: mode-specific max_tokens
# ---------------------------------------------------------------------------
MODE_MAX_TOKENS = {
    "governing_reference": 64,
    "numeric_lookup": 80,
    "extractive_definition": 128,
    "requirement_list": 160,
    "multi_aspect": 192,
    "process_workflow": 192,
}

# ---------------------------------------------------------------------------
# Query complexity
# ---------------------------------------------------------------------------
SIMPLE_PATTERNS = [
    r"^what\s+is\s+the\b",
    r"^what\s+are\s+the\s+requirements?\s+for\b",
    r"^what\s+is\s+required\b",
    r"^how\s+many\b",
    r"^what\s+size\b",
    r"^what\s+type\b",
]
COMPLEX_PATTERNS = [
    r"compare", r"difference\s+between", r"how\s+does.*interact",
    r"relationship\s+between", r"impact\s+of.*on", r"trade-?off",
    r"explain\s+why", r"what\s+happens\s+if",
]


def classify_query_complexity(query):
    q = query.lower().strip()
    for p in COMPLEX_PATTERNS:
        if re.search(p, q):
            return "complex"
    for p in SIMPLE_PATTERNS:
        if re.search(p, q):
            return "simple"
    return "moderate"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.expanduser("~/marley1/compression/.cache")

def cache_key(doc_path, query):
    h = hashlib.sha256(f"{doc_path}::{query}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.json")

def cache_get(doc_path, query):
    path = cache_key(doc_path, query)
    if os.path.exists(path):
        with open(path) as f:
            entry = json.load(f)
        if time.time() - entry.get("timestamp", 0) < 86400:
            return entry
    return None

def cache_set(doc_path, query, answer, metadata):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_key(doc_path, query), "w") as f:
        json.dump({"query": query, "doc": doc_path, "answer": answer,
                    "metadata": metadata, "timestamp": time.time()}, f, indent=2)


def check_fatman(url="http://100.97.87.86:8080/health", timeout=5):
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def route(doc_text, query, doc_path=None):
    t0 = time.perf_counter()
    decisions = {}

    if doc_path:
        cached = cache_get(doc_path, query)
        if cached:
            decisions["cached_answer"] = cached["answer"]
            decisions["cache_hit"] = True
            decisions["route_time"] = time.perf_counter() - t0
            return decisions
    decisions["cache_hit"] = False

    sample = doc_text[:5000]
    scores = {}
    for dtype, config in DOC_SIGNATURES.items():
        score = sum(1 for p in config["patterns"] if re.search(p, sample, re.IGNORECASE))
        if score > 0:
            scores[dtype] = score

    if scores:
        doc_type = max(scores, key=scores.get)
        decisions["doc_type"] = doc_type
        decisions["doc_type_confidence"] = scores[doc_type]
        decisions["codebook"] = DOC_SIGNATURES[doc_type]["codebook"]
        decisions["system_prompt"] = DOC_SIGNATURES[doc_type]["system_prompt"]
    else:
        decisions["doc_type"] = "unknown"
        decisions["doc_type_confidence"] = 0
        decisions["codebook"] = "construction"
        decisions["system_prompt"] = "You are a document analysis expert. Be specific." + _GROUNDING

    # Answer mode with hard overrides
    answer_mode = classify_answer_mode(query)
    decisions["answer_mode"] = answer_mode

    # Query expansion (capped for governing_reference)
    expanded = expand_query(query)
    decisions["expanded_query"] = expanded
    decisions["query_expanded"] = expanded != query
    decisions["cap_expansion"] = answer_mode == "governing_reference"

    # Chunk/topk tuning
    token_est = len(doc_text.split())
    if token_est < 5000:
        decisions["chunk_size"] = 3
        decisions["top_k"] = 5
    elif token_est < 20000:
        decisions["chunk_size"] = 5
        decisions["top_k"] = 10
    elif token_est < 50000:
        decisions["chunk_size"] = 5
        decisions["top_k"] = 10
    else:
        decisions["chunk_size"] = 7
        decisions["top_k"] = 10

    # Max tokens from mode, not complexity
    decisions["max_tokens"] = MODE_MAX_TOKENS.get(answer_mode, 160)
    decisions["query_complexity"] = classify_query_complexity(query)

    # Context budget guard
    context_budget = 8192 - 300 - decisions["max_tokens"]
    est_tokens = decisions["top_k"] * decisions["chunk_size"] * 20
    while est_tokens > context_budget and decisions["top_k"] > 3:
        decisions["top_k"] -= 1
        est_tokens = decisions["top_k"] * decisions["chunk_size"] * 20

    decisions["skip_abbrev"] = decisions["codebook"] == "construction"
    decisions["fatman_healthy"] = check_fatman()
    decisions["route_time"] = time.perf_counter() - t0
    return decisions


if __name__ == "__main__":
    test_queries = [
        ("What are the progressive collapse prevention requirements?", "governing_reference"),
        ("What are the blast resistance requirements?", "governing_reference"),
        ("What are the storage and closet space requirements?", "numeric_lookup"),
        ("What are the laundry facility requirements?", "numeric_lookup"),
        ("What are the cybersecurity requirements for microgrid systems?", "requirement_list"),
        ("What are the islanding requirements for military microgrids?", "requirement_list"),
        ("What are the metering and monitoring requirements?", "requirement_list"),
        ("What are the antenna and RF system requirements?", "requirement_list"),
    ]
    print("Answer-mode classification tests:")
    for q, expected in test_queries:
        mode = classify_answer_mode(q)
        status = "OK" if mode == expected else f"FAIL (got {mode})"
        print(f"  [{mode:25s}] {status:10s} {q}")
