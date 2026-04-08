#!/usr/bin/env python3
"""Marley1 Compression Pipeline — PDF edition.

Extracts text from a PDF, then runs the full compression stack:
  RelevanceFilter -> AbbrevCompressor -> LLMLingua-2 -> Fat Man inference
"""

import json
import sys
import os
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pdfplumber
from relevance_filter import RelevanceFilter
from abbrev import ConstructionCompressor
#from llmlingua import PromptCompressor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_PATH = os.path.expanduser("~/marley1/specs/ufc_unaccompanied_housing.pdf")
FAT_MAN_URL = "http://192.168.88.15:8080/v1/chat/completions"
QUERY = "What are the design standards for military unaccompanied housing?"

CHUNK_SIZE = 5          # sentences per chunk (larger doc = larger chunks)
RELEVANCE_TOP_K = 10
SKIP_RELEVANCE = True
LINGUA_RATE = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def word_count(text):
    return len(text.split())


def pct(part, whole):
    return (1 - part / whole) * 100 if whole else 0


def stage_header(num, name):
    print("\n" + "=" * 70)
    print("  STAGE {}: {}".format(num, name))
    print("=" * 70)


def stage_report(text, prev_chars, orig_chars, elapsed):
    chars = len(text)
    tokens = word_count(text)
    print("  Chars          : {:,}".format(chars))
    print("  Tokens (approx): {:,}".format(tokens))
    print("  Stage reduction : {:.1f}%".format(pct(chars, prev_chars)))
    print("  Cumulative      : {:.1f}% from original".format(pct(chars, orig_chars)))
    print("  Time            : {:.2f}s".format(elapsed))
    return chars


def chat_completion(user_content, system_content):
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
    print("  MARLEY1 PDF COMPRESSION PIPELINE")
    print("=" * 70)
    print("  PDF      : {}".format(PDF_PATH))
    print("  Query    : \"{}\"".format(QUERY))
    print("  Target   : Fat Man (Qwen2.5-3B @ {})".format(FAT_MAN_URL))
    print("  Chunks   : {} sentences, top_k={}".format(CHUNK_SIZE, RELEVANCE_TOP_K))
    print("  LLMLingua: rate={}".format(LINGUA_RATE))
    print("=" * 70)

    # ------------------------------------------------------------------
    # Stage 0: PDF Text Extraction
    # ------------------------------------------------------------------
    stage_header(0, "PDF TEXT EXTRACTION (pdfplumber)")
    t0 = time.perf_counter()
    pdf = pdfplumber.open(PDF_PATH)
    page_count = len(pdf.pages)
    print("  Pages: {}".format(page_count))

    page_texts = []
    for p in pdf.pages:
        t = p.extract_text() or ""
        page_texts.append(t)
    pdf.close()

    full_text = "\n".join(page_texts)
    t_extract = time.perf_counter() - t0

    orig_chars = len(full_text)
    orig_tokens = word_count(full_text)
    print("  Extracted {:,} chars, ~{:,} tokens".format(orig_chars, orig_tokens))
    print("  Time: {:.1f}s".format(t_extract))

    # ------------------------------------------------------------------
    # Stage 1: Relevance Filter
    # ------------------------------------------------------------------
    stage_header(1, "RELEVANCE FILTER (all-MiniLM-L6-v2)")
    print("  Loading embedding model...")
    t0 = time.perf_counter()
    rf = RelevanceFilter()
    t_model_load = time.perf_counter() - t0
    print("  Model loaded in {:.1f}s".format(t_model_load))

    t0 = time.perf_counter()
    chunks = rf.chunk(full_text, chunk_size=CHUNK_SIZE)
    print("  Chunked into {} pieces ({} sentences each)".format(len(chunks), CHUNK_SIZE))

    if SKIP_RELEVANCE:
        kept_indices = list(range(len(chunks)))
        similarities = [1.0] * len(chunks)
    else:
        kept_indices, similarities = rf.filter(chunks, QUERY, top_k=RELEVANCE_TOP_K)
    t_filter = time.perf_counter() - t0

    filtered_text = " ".join(chunks[int(i)] for i in kept_indices)

    print("\n  Kept chunks:")
    for idx in kept_indices:
        i = int(idx)
        preview = chunks[i][:80].replace("\n", " ")
        print("    #{:4d}  sim={:.3f}  \"{}...\"".format(i, similarities[i], preview))

    prev_chars = stage_report(filtered_text, orig_chars, orig_chars, t_filter)

    # ------------------------------------------------------------------
    # Stage 2: Abbreviation Compressor
    # ------------------------------------------------------------------
    stage_header(2, "ABBREVIATION COMPRESSOR (abbrev.py)")
    t0 = time.perf_counter()
    abbrev = ConstructionCompressor()
    abbrev_text = abbrev.compress(filtered_text)
    t_abbrev = time.perf_counter() - t0

    prev_chars = stage_report(abbrev_text, prev_chars, orig_chars, t_abbrev)

    # ------------------------------------------------------------------
    # Stage 3: LLMLingua-2
    # ------------------------------------------------------------------
#    stage_header(3, "LLMLINGUA-2 (xlm-roberta-large, rate={})".format(LINGUA_RATE))
#    print("  Loading LLMLingua-2 model...")
#    t0 = time.perf_counter()
#    lingua = PromptCompressor(
#        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
#        use_llmlingua2=True,
#        device_map="cpu",
#    )
#    print("  Model loaded in {:.1f}s".format(time.perf_counter() - t0))
#
#    t0 = time.perf_counter()
#    result = lingua.compress_prompt(
#        [abbrev_text],
#        
#        force_tokens=['\n', '.', ',', '?', '!'],
#    )
#    0 = time.perf_counter() - t0
#
#    abbrev_text = result["compressed_prompt"]
#    lingua_orig_tok = result.get("origin_tokens", word_count(abbrev_text))
#    lingua_comp_tok = result.get("compressed_tokens", word_count(abbrev_text))
#
#    prev_chars = stage_report(abbrev_text, prev_chars, orig_chars, 0)
#    print("  LLMLingua tokens: {} -> {}".format(lingua_orig_tok, lingua_comp_tok))
#
#    print("\n  Compressed text preview (first 500 chars):")
#    print("  " + abbrev_text[:500].replace("\n", "\n  "))
#    if len(abbrev_text) > 500:
    #        print("  ...")

    # ------------------------------------------------------------------
    # Stage 4: Fat Man Inference
    # ------------------------------------------------------------------
    stage_header(4, "FAT MAN INFERENCE (Qwen2.5-3B)")

    system_prompt = (
        "You are a scheduling software expert. The user will ask a question "
        "about NetPoint scheduling software. Answer based on the provided "
        "context. The text may contain abbreviations. Give a clear, "
        "step-by-step answer."
    )
    user_prompt = "Question: {}\n\nContext:\n{}".format(QUERY, abbrev_text)

    print("  Sending to {}...".format(FAT_MAN_URL))
    print("  System prompt  : {} chars".format(len(system_prompt)))
    print("  User prompt    : {} chars".format(len(user_prompt)))
    print("  Final tokens   : ~{}".format(word_count(user_prompt)))

    t0 = time.perf_counter()
    response = chat_completion(user_prompt, system_prompt)
    t_inference = time.perf_counter() - t0

    print("  Prompt tokens  : {}".format(response["prompt_tokens"]))
    print("  Completion tok : {}".format(response["completion_tokens"]))
    print("  Inference time : {:.2f}s".format(t_inference))

    print("\n  FAT MAN RESPONSE:")
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

    stages = [
        ("Original PDF text", orig_chars, orig_tokens),
        ("After relevance filter", len(filtered_text), word_count(filtered_text)),
        ("After abbreviation", len(abbrev_text), word_count(abbrev_text)),
        ("After LLMLingua-2", len(abbrev_text), word_count(abbrev_text)),
        ("Sent to Fat Man", len(user_prompt), response["prompt_tokens"]),
    ]

    print("  {:30s} {:>10s}  {:>10s}  {:>10s}".format(
        "Stage", "Chars", "Tokens", "Cumul %"))
    print("  " + "-" * 64)
    for name, chars, tokens in stages:
        cumul = pct(chars, orig_chars)
        print("  {:30s} {:>10,}  {:>10,}  {:>9.1f}%".format(
            name, chars, tokens, cumul))

    print("  " + "-" * 64)
    print("  Total compression: {:,} chars -> {:,} chars ({:.1f}x)".format(
        orig_chars, len(abbrev_text), orig_chars / len(abbrev_text) if len(abbrev_text) else 0))
    print("  Total tokens    : ~{:,} -> {} prompt tokens ({:.0f}x)".format(
        orig_tokens, response["prompt_tokens"],
        orig_tokens / response["prompt_tokens"] if response["prompt_tokens"] else 0))
    print("  Total pipeline time: {:.1f}s".format(t_total))
    print("  (extract: {:.1f}s | relevance: {:.1f}s | abbrev: {:.2f}s | "
          "lingua: {:.1f}s | inference: {:.1f}s)".format(
              t_extract, t_filter, t_abbrev, 0, t_inference))
    print("=" * 70)


if __name__ == "__main__":
    main()
