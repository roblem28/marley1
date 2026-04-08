#!/usr/bin/env python3
"""Marley1 Compression Pipeline.

Full stack: RelevanceFilter -> AbbrevCompressor -> LLMLingua-2 -> Fat Man inference.

Each stage prints char count, token estimate, stage reduction, and cumulative
reduction from the original document.
"""

import json
import sys
import os
import time
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relevance_filter import RelevanceFilter
from abbrev import ConstructionCompressor
from llmlingua import PromptCompressor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FAT_MAN_URL = "http://192.168.88.15:8080/v1/chat/completions"
LINGUA_RATE = 0.5

# ---------------------------------------------------------------------------
# 50-sentence construction spec (same as relevance_filter.py)
# ---------------------------------------------------------------------------
SPEC_DOC = (
    # Site Work (sentences 1-8)
    "The site contractor shall complete all earthwork and grading operations prior to foundation work. "
    "Topsoil shall be stripped to a depth of 12 inches and stockpiled on site for later use. "
    "Erosion control measures including silt fencing and sediment basins shall be installed per the SWPPP. "
    "The geotechnical engineer has approved the bearing capacity at 3000 PSF for spread footings. "
    "All underground utilities shall be located and marked before excavation begins. "
    "Temporary construction roads shall use 6 inches of compacted gravel base. "
    "Dewatering operations are required in the northwest corner due to high water table conditions. "
    "The site lighting plan includes 20-foot pole-mounted LED fixtures at 50-foot spacing along access roads. "
    # Structural (sentences 9-16)
    "Structural steel erection shall proceed from grid A to grid J in sequence. "
    "All moment connections require CJP welds with ultrasonic testing per AWS D1.1. "
    "The precast concrete panels for the building envelope weigh approximately 8 tons each. "
    "Anchor bolt placement tolerances shall not exceed 1/8 inch from plan location. "
    "The raised access floor system requires a minimum 18-inch plenum depth. "
    "Seismic bracing for all equipment over 400 pounds shall be designed per ASCE 7. "
    "The roof structure consists of open web steel joists at 5-foot spacing with metal deck. "
    "Fireproofing of structural steel shall achieve a 2-hour fire rating per UL assembly. "
    # HVAC (sentences 17-26)
    "The HVAC system design includes four rooftop units serving the office areas. "
    "Each rooftop unit is rated at 25 tons cooling capacity with gas heat. "
    "Variable air volume boxes with DDC controls serve all interior zones. "
    "The commissioning agent shall verify proper airflow at each VAV terminal. "
    "HVAC commissioning requirements include functional performance testing of all air handling units. "
    "The building automation system shall demonstrate trending and alarm capabilities during commissioning. "
    "Ductwork leakage testing shall not exceed 4 CFM per 100 square feet of duct surface area. "
    "Refrigerant piping shall be pressure tested at 350 PSI for 24 hours with nitrogen. "
    "The energy recovery ventilator shall achieve a minimum 72 percent effectiveness rating. "
    "HVAC controls sequences of operation shall be reviewed and approved prior to commissioning startup. "
    # Electrical (sentences 27-36)
    "The main electrical service is 4000 amps at 480/277 volts three-phase. "
    "Two 2000 KVA dry-type transformers feed the main distribution switchgear. "
    "The emergency generator is rated at 1500 KW with a 500-gallon belly tank. "
    "Automatic transfer switches shall be tested monthly with load bank verification annually. "
    "The lighting control system uses DALI protocol with daylight harvesting in perimeter zones. "
    "All branch circuit wiring in data center areas shall be installed in cable tray. "
    "The grounding system includes a ground ring with 20-foot driven rods at building corners. "
    "Arc flash labels shall be applied to all panels and switchgear per NFPA 70E. "
    "The electrical contractor shall provide coordination study results before energization. "
    "Temporary power distribution shall maintain 200 amps per floor during construction. "
    # Plumbing (sentences 37-42)
    "Domestic water service enters the building through a 4-inch copper main with backflow preventer. "
    "The hot water system uses two 100-gallon commercial water heaters in parallel configuration. "
    "Sanitary sewer connects to the municipal system via a 6-inch PVC lateral at the property line. "
    "Roof drainage includes internal conductors sized for a 100-year storm event. "
    "All plumbing fixtures shall be low-flow models meeting WaterSense certification requirements. "
    "The grease interceptor for the kitchen area is sized at 50 GPM with 100 pounds capacity. "
    # Fire Protection (sentences 43-50)
    "The fire suppression system is a wet pipe sprinkler design per NFPA 13. "
    "Sprinkler coverage in the data center uses a pre-action dry pipe system with double interlock. "
    "The fire alarm control panel is an addressable system with voice evacuation capability. "
    "Smoke detectors are required above and below the raised access floor in server rooms. "
    "Fire dampers shall be installed at all rated wall and floor penetrations. "
    "The fire pump is a 750 GPM electric-driven unit with jockey pump and controller. "
    "Standpipe connections are required at each stairwell landing per the fire code. "
    "The clean agent suppression system in the MDF uses FM-200 with a 10-second discharge time."
)

QUERY = "What are the HVAC commissioning requirements?"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def word_count(text):
    """Approximate token count via whitespace split."""
    return len(text.split())


def stage_banner(stage_num, name):
    print("\n" + "=" * 70)
    print("  STAGE {}: {}".format(stage_num, name))
    print("=" * 70)


def stage_metrics(label, text, prev_chars, orig_chars, elapsed):
    """Print standardized metrics for a pipeline stage."""
    chars = len(text)
    tokens = word_count(text)
    stage_pct = (1 - chars / prev_chars) * 100 if prev_chars else 0
    cumul_pct = (1 - chars / orig_chars) * 100 if orig_chars else 0
    print("  Chars          : {:,}".format(chars))
    print("  Tokens (approx): {:,}".format(tokens))
    print("  Stage reduction : {:.1f}%".format(stage_pct))
    print("  Cumulative      : {:.1f}% from original".format(cumul_pct))
    print("  Time            : {:.2f}s".format(elapsed))
    return chars


def chat_completion(user_content, system_content):
    """Send request to Fat Man and return response text + timing."""
    body = {
        "model": "qwen2.5-3b",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "max_tokens": 512,
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        FAT_MAN_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode("utf-8")
    elapsed = time.perf_counter() - t0

    obj = json.loads(raw)
    usage = obj.get("usage", {})
    reply = obj["choices"][0]["message"]["content"]

    return {
        "reply": reply,
        "elapsed": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    t_pipeline = time.perf_counter()

    print("=" * 70)
    print("  MARLEY1 COMPRESSION PIPELINE")
    print("=" * 70)
    print("  Document   : 50-sentence construction spec")
    print("  Query      : \"{}\"".format(QUERY))
    print("  Target     : Fat Man (Qwen2.5-3B @ {})".format(FAT_MAN_URL))
    print("  LLMLingua  : rate={}".format(LINGUA_RATE))
    print("=" * 70)

    orig_chars = len(SPEC_DOC)
    orig_tokens = word_count(SPEC_DOC)
    print("\n  ORIGINAL DOCUMENT")
    print("  Chars  : {:,}".format(orig_chars))
    print("  Tokens : {:,}".format(orig_tokens))

    # ------------------------------------------------------------------
    # Stage 1: Relevance Filter
    # ------------------------------------------------------------------
    stage_banner(1, "RELEVANCE FILTER (all-MiniLM-L6-v2)")
    print("  Loading embedding model...")
    t0 = time.perf_counter()
    rf = RelevanceFilter()
    print("  Model loaded in {:.1f}s".format(time.perf_counter() - t0))

    chunks = rf.chunk(SPEC_DOC, chunk_size=3)
    print("  Chunked into {} pieces (3 sentences each)".format(len(chunks)))

    t0 = time.perf_counter()
    kept_indices, similarities = rf.filter(chunks, QUERY, top_k=3)
    t_filter = time.perf_counter() - t0

    filtered_text = " ".join(chunks[int(i)] for i in kept_indices)

    print("\n  Kept chunks: {}".format(
        ["#{} (sim={:.3f})".format(int(i), similarities[int(i)]) for i in kept_indices]))
    prev_chars = stage_metrics("relevance", filtered_text, orig_chars, orig_chars, t_filter)

    print("\n  Text:")
    print("  " + filtered_text[:300])
    if len(filtered_text) > 300:
        print("  ...")

    # ------------------------------------------------------------------
    # Stage 2: Abbreviation Compressor
    # ------------------------------------------------------------------
    stage_banner(2, "ABBREVIATION COMPRESSOR (abbrev.py)")
    t0 = time.perf_counter()
    abbrev = ConstructionCompressor()
    abbrev_text = abbrev.compress(filtered_text)
    t_abbrev = time.perf_counter() - t0

    prev_chars = stage_metrics("abbrev", abbrev_text, prev_chars, orig_chars, t_abbrev)

    print("\n  Text:")
    print("  " + abbrev_text[:300])
    if len(abbrev_text) > 300:
        print("  ...")

    # ------------------------------------------------------------------
    # Stage 3: LLMLingua-2
    # ------------------------------------------------------------------
    stage_banner(3, "LLMLINGUA-2 (xlm-roberta-large, rate={})".format(LINGUA_RATE))
    print("  Loading LLMLingua-2 model...")
    t0 = time.perf_counter()
    lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
        device_map="cpu",
    )
    print("  Model loaded in {:.1f}s".format(time.perf_counter() - t0))

    t0 = time.perf_counter()
    result = lingua.compress_prompt(
        [abbrev_text],
        rate=LINGUA_RATE,
        force_tokens=['\n', '.', ',', '?', '!'],
    )
    t_lingua = time.perf_counter() - t0

    lingua_text = result["compressed_prompt"]
    lingua_orig_tok = result.get("origin_tokens", word_count(abbrev_text))
    lingua_comp_tok = result.get("compressed_tokens", word_count(lingua_text))

    prev_chars = stage_metrics("lingua", lingua_text, prev_chars, orig_chars, t_lingua)
    print("  LLMLingua tokens: {} -> {}".format(lingua_orig_tok, lingua_comp_tok))

    print("\n  Text:")
    print("  " + lingua_text)

    # ------------------------------------------------------------------
    # Stage 4: Fat Man Inference
    # ------------------------------------------------------------------
    stage_banner(4, "FAT MAN INFERENCE (Qwen2.5-3B)")

    system_prompt = (
        "You are a construction industry assistant. The text may contain "
        "standard construction abbreviations (HVAC, VAV, DDC, BAS, AHU, "
        "CX = commissioning, CFM, PSI, etc). "
        "Answer the question based on the provided text."
    )
    user_prompt = "Question: {}\n\nContext:\n{}".format(QUERY, lingua_text)

    print("  Sending to {}...".format(FAT_MAN_URL))
    print("  System prompt : {} chars".format(len(system_prompt)))
    print("  User prompt   : {} chars".format(len(user_prompt)))

    t0 = time.perf_counter()
    response = chat_completion(user_prompt, system_prompt)
    t_inference = time.perf_counter() - t0

    print("  Prompt tokens  : {}".format(response["prompt_tokens"]))
    print("  Completion tok : {}".format(response["completion_tokens"]))
    print("  Inference time : {:.2f}s".format(t_inference))

    print("\n  MODEL RESPONSE:")
    print("  " + "-" * 60)
    for line in response["reply"].split("\n"):
        print("  " + line)
    print("  " + "-" * 60)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    t_total = time.perf_counter() - t_pipeline

    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    print("  {:30s} {:>8s}  {:>8s}  {:>10s}".format(
        "Stage", "Chars", "Tokens", "Cumul %"))
    print("  " + "-" * 60)

    stages = [
        ("Original document", orig_chars, orig_tokens, 0),
        ("After relevance filter", len(filtered_text), word_count(filtered_text),
         (1 - len(filtered_text) / orig_chars) * 100),
        ("After abbreviation", len(abbrev_text), word_count(abbrev_text),
         (1 - len(abbrev_text) / orig_chars) * 100),
        ("After LLMLingua-2", len(lingua_text), word_count(lingua_text),
         (1 - len(lingua_text) / orig_chars) * 100),
    ]

    for name, chars, tokens, pct in stages:
        print("  {:30s} {:>8,}  {:>8,}  {:>9.1f}%".format(name, chars, tokens, pct))

    print("  " + "-" * 60)
    print("  Total pipeline time: {:.2f}s".format(t_total))
    print("  (relevance: {:.2f}s | abbrev: {:.2f}s | lingua: {:.2f}s | inference: {:.2f}s)".format(
        t_filter, t_abbrev, t_lingua, t_inference))
    print("=" * 70)


if __name__ == "__main__":
    main()
