#!/usr/bin/env python3
"""Marley1 Neuro-Symbolic Query Router.

Symbolic decision layer that runs before any neural compute.
Decides: codebook, chunk_size, top_k, max_tokens, system prompt.
Zero GPU cost. Pure rules.
"""

import os
import re
import time
import hashlib
import json

# ---------------------------------------------------------------------------
# Document type detection (symbolic — pattern matching, no model)
# ---------------------------------------------------------------------------
DOC_SIGNATURES = {
    "electrical": {
        "patterns": [r"UFC\s+3-501", r"ELECTRICAL\s+ENGINEERING", r"electrical\s+design", r"grounding", r"switchgear", r"NFPA\s+70", r"arc\s+flash", r"transformer", r"circuit"],
        "codebook": "construction",
        "system_prompt": "You are a DoD electrical systems expert. Answer based on the provided context only. Be specific and cite section numbers when available.",
    },
    "structural": {
        "patterns": [r"UFC\s+3-301", r"STRUCTURAL\s+ENGINEERING", r"seismic", r"wind\s+load", r"ASCE\s+7", r"progressive\s+collapse", r"foundation"],
        "codebook": "construction",
        "system_prompt": "You are a DoD structural engineering expert. Answer based on the provided context only. Be specific about load requirements and design criteria.",
    },
    "mechanical": {
        "patterns": [r"HVAC", r"rooftop\s+unit", r"air\s+handling", r"ductwork",
                     r"commissioning", r"VAV", r"chiller", r"boiler"],
        "codebook": "construction",
        "system_prompt": "You are a DoD mechanical/HVAC systems expert. Answer based on the provided context only. Include equipment specs and performance criteria.",
    },
    "plumbing": {
        "patterns": [r"plumbing", r"domestic\s+water", r"sanitary", r"backflow",
                     r"water\s+heater", r"drainage", r"GPM"],
        "codebook": "construction",
        "system_prompt": "You are a DoD plumbing systems expert. Answer based on the provided context only. Be specific about sizing and code requirements.",
    },
    "fire_protection": {
        "patterns": [r"fire\s+suppression", r"sprinkler", r"NFPA\s+13", r"fire\s+alarm",
                     r"pre-action", r"standpipe", r"fire\s+pump"],
        "codebook": "construction",
        "system_prompt": "You are a DoD fire protection expert. Answer based on the provided context only. Reference NFPA standards when applicable.",
    },
    "c5isr": {
        "patterns": [r"UFC\s+4-141", r"C5ISR", r"SURVEILLANCE", r"RECONNAISSANCE", r"COMMAND.*CONTROL", r"antenna", r"telecommunications"],
        "codebook": "military",
        "system_prompt": "You are a DoD C5ISR facilities expert. Answer based on the provided context only. Be specific about infrastructure and security requirements.",
    },
    "housing": {
        "patterns": [r"unaccompanied\s+housing", r"barracks", r"living\s+area",
                     r"bedroom", r"common\s+area", r"laundry", r"ADA\s+compliance"],
        "codebook": "construction",
        "system_prompt": "You are a DoD housing design expert. Answer based on the provided context only. Be specific about space requirements and amenity standards.",
    },
    "microgrid": {
        "patterns": [r"microgrid", r"islanding", r"distributed\s+energy",
                     r"energy\s+storage", r"grid\s+tie", r"renewable"],
        "codebook": "construction",
        "system_prompt": "You are a DoD microgrid and power systems expert. Answer based on the provided context only. Be specific about control requirements and redundancy.",
    },
    "medical": {
        "patterns": [r"patient", r"diagnosis", r"treatment", r"clinical",
                     r"dosage", r"prognosis", r"ICD-10"],
        "codebook": "medical",
        "system_prompt": "You are a medical document analysis expert. Answer based on the provided context only. Be precise with clinical terminology.",
    },
    "legal": {
        "patterns": [r"plaintiff", r"defendant", r"jurisdiction", r"statute",
                     r"tort", r"liability", r"contract\s+law"],
        "codebook": "legal",
        "system_prompt": "You are a legal document analysis expert. Answer based on the provided context only. Reference specific clauses and sections.",
    },
    "financial": {
        "patterns": [r"revenue", r"EBITDA", r"balance\s+sheet", r"SEC\s+filing",
                     r"quarterly\s+report", r"P/?E\s+ratio", r"fiscal\s+year"],
        "codebook": "financial",
        "system_prompt": "You are a financial document analysis expert. Answer based on the provided context only. Be precise with figures and reporting periods.",
    },
}

# ---------------------------------------------------------------------------
# Query complexity detection (symbolic — heuristics, no model)
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
    r"compare",
    r"difference\s+between",
    r"how\s+does.*interact",
    r"relationship\s+between",
    r"impact\s+of.*on",
    r"trade-?off",
    r"explain\s+why",
    r"what\s+happens\s+if",
]


def classify_query_complexity(query):
    """Returns 'simple', 'moderate', or 'complex'."""
    q = query.lower().strip()
    for p in COMPLEX_PATTERNS:
        if re.search(p, q):
            return "complex"
    for p in SIMPLE_PATTERNS:
        if re.search(p, q):
            return "simple"
    return "moderate"


# ---------------------------------------------------------------------------
# Response cache (symbolic — hash-based, no model)
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
        age = time.time() - entry.get("timestamp", 0)
        if age < 86400:  # 24h TTL
            return entry
    return None


def cache_set(doc_path, query, answer, metadata):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_key(doc_path, query)
    with open(path, "w") as f:
        json.dump({
            "query": query,
            "doc": doc_path,
            "answer": answer,
            "metadata": metadata,
            "timestamp": time.time(),
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Health check (symbolic — ping, no model)
# ---------------------------------------------------------------------------
def check_fatman(url="http://100.97.87.86:8080/health", timeout=5):
    """Returns True if Fat Man is responding."""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Router — the main symbolic decision layer
# ---------------------------------------------------------------------------
def route(doc_text, query, doc_path=None):
    """
    Makes all decisions before any neural compute.

    Returns a dict of pipeline parameters:
      codebook, chunk_size, top_k, max_tokens, system_prompt,
      skip_abbrev, cached_answer, fatman_healthy
    """
    t0 = time.perf_counter()
    decisions = {}

    # 1. Check cache
    if doc_path:
        cached = cache_get(doc_path, query)
        if cached:
            decisions["cached_answer"] = cached["answer"]
            decisions["cache_hit"] = True
            decisions["route_time"] = time.perf_counter() - t0
            return decisions
    decisions["cache_hit"] = False

    # 2. Detect document type
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
        decisions["codebook"] = "construction"  # default
        decisions["system_prompt"] = "You are a document analysis expert. Answer based on the provided context only. Be specific."

    # 3. Size-based chunk/topk tuning
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

    # 4. Query complexity -> max_tokens
    complexity = classify_query_complexity(query)
    decisions["query_complexity"] = complexity
    if complexity == "simple":
        decisions["max_tokens"] = 256
    elif complexity == "moderate":
        decisions["max_tokens"] = 512
    else:
        decisions["max_tokens"] = 768

    # 4b. Context budget guard (8192 total - system prompt ~200 - formatting ~100)
    context_budget = 8192 - 300 - decisions["max_tokens"]
    # Rough estimate: top_k chunks * chunk_size sentences * ~20 tokens/sentence
    est_tokens = decisions["top_k"] * decisions["chunk_size"] * 20
    while est_tokens > context_budget and decisions["top_k"] > 3:
        decisions["top_k"] -= 1
        est_tokens = decisions["top_k"] * decisions["chunk_size"] * 20

    # 5. Skip abbreviation if codebook is unlikely to help
    decisions["skip_abbrev"] = decisions["codebook"] == "construction"  # marginal on gov prose

    # 6. Fat Man health
    decisions["fatman_healthy"] = check_fatman()

    decisions["route_time"] = time.perf_counter() - t0
    return decisions


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pdfplumber

    # Test with a real doc
    specs_dir = os.path.expanduser("~/marley1/specs")
    test_files = [f for f in os.listdir(specs_dir) if f.endswith(".pdf")]

    if not test_files:
        print("No PDFs in ~/marley1/specs/")
        exit(1)

    test_pdf = os.path.join(specs_dir, test_files[0])
    print(f"Testing router with: {test_pdf}")

    pdf = pdfplumber.open(test_pdf)
    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    pdf.close()

    queries = [
        "What are the electrical design requirements?",
        "Compare the seismic and wind load requirements.",
        "What size is the emergency generator?",
        "What happens if the microgrid loses grid connection?",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {q}")
        result = route(text, q, test_pdf)
        for k, v in result.items():
            print(f"  {k:25s}: {v}")
