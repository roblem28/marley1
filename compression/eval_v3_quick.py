#!/usr/bin/env python3
"""Quick eval v3: 10 hardest queries through updated pipeline.

Changes from v2:
- Router system prompts now include grounding instruction
- RelevanceFilterV2 now has requirement extraction + table parsing
- Saves to answers_v3_quick.json
"""
import json, sys, os, time, urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pdfplumber
from router import route
from relevance_filter_v2 import RelevanceFilterV2

FAT_MAN_URL = "http://100.97.87.86:8080/v1/chat/completions"
FAT_MAN_HEALTH = "http://100.97.87.86:8080/health"
SPECS_DIR = Path(os.path.expanduser("~/marley1/specs"))
OUT_DIR = Path(os.path.expanduser("~/marley1/compression"))

HARD_CASES = [
    {"doc": "ufc_electrical", "query": "What are the metering and monitoring requirements?"},
    {"doc": "ufc_electrical", "query": "What are the requirements for renewable energy integration?"},
    {"doc": "ufc_structural", "query": "What are the progressive collapse prevention requirements?"},
    {"doc": "ufc_structural", "query": "What are the blast resistance requirements?"},
    {"doc": "ufc_microgrid", "query": "What are the cybersecurity requirements for microgrid systems?"},
    {"doc": "ufc_microgrid", "query": "What are the islanding requirements for military microgrids?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the power and grounding requirements for C5ISR systems?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the antenna and RF system requirements?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the laundry facility requirements?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the storage and closet space requirements?"},
]


def check_fatman(timeout=5):
    try:
        with urllib.request.urlopen(FAT_MAN_HEALTH, timeout=timeout) as r:
            return r.status == 200
    except:
        return False


def wait_for_fatman(max_wait=120):
    if check_fatman():
        return True
    print("  Fat Man unresponsive, waiting...")
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(5)
        if check_fatman():
            print(f"  Fat Man recovered after {time.time()-t0:.0f}s")
            return True
    print(f"  Fat Man did not recover after {max_wait}s")
    return False


def chat(messages, system=None, max_tokens=512, temp=0.1):
    body = {"model": "qwen2.5-3b", "messages": messages, "max_tokens": max_tokens, "temperature": temp}
    if system:
        body["messages"] = [{"role": "system", "content": system}] + messages
    data = json.dumps(body).encode()
    req = urllib.request.Request(FAT_MAN_URL, data, {"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                obj = json.loads(r.read())
            return obj["choices"][0]["message"]["content"], obj.get("usage", {})
        except Exception as e:
            if attempt < 2:
                print(f"  Retry {attempt+1}/2: {e}")
                if not wait_for_fatman(60):
                    raise
            else:
                raise


def extract_pdf(path):
    pdf = pdfplumber.open(path)
    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    pdf.close()
    return text


def main():
    print(f"\n{'='*60}")
    print("EVAL V3 QUICK — grounded prompts + req extraction + tables")
    print(f"{'='*60}")

    doc_names = sorted(set(c["doc"] for c in HARD_CASES))
    doc_texts = {}
    doc_tokens = {}

    print("\nExtracting PDFs...")
    for doc_name in doc_names:
        pdf_path = SPECS_DIR / f"{doc_name}.pdf"
        t0 = time.perf_counter()
        text = extract_pdf(pdf_path)
        elapsed = time.perf_counter() - t0
        tokens = len(text.split())
        doc_texts[doc_name] = text
        doc_tokens[doc_name] = tokens
        print(f"  {doc_name}: {tokens:,} tokens in {elapsed:.1f}s")

    print("\nLoading RelevanceFilterV2 (bi-encoder + cross-encoder)...")
    t0 = time.perf_counter()
    rf = RelevanceFilterV2()
    print(f"  Models loaded in {time.perf_counter() - t0:.1f}s")

    print("\nSection-aware chunking (with table parsing) and embedding...")
    doc_chunks = {}
    doc_headers = {}
    doc_embeddings = {}

    for doc_name in doc_names:
        t0 = time.perf_counter()
        chunks, headers = rf.chunk_with_headers(doc_texts[doc_name])
        elapsed_chunk = time.perf_counter() - t0

        n_tables = sum(1 for h in headers if h.startswith("Table"))
        n_sections = len(headers) - n_tables

        t0 = time.perf_counter()
        embs = rf.precompute(chunks)
        elapsed_emb = time.perf_counter() - t0

        doc_chunks[doc_name] = chunks
        doc_headers[doc_name] = headers
        doc_embeddings[doc_name] = embs
        print(f"  {doc_name}: {n_sections} sections + {n_tables} table rows, "
              f"chunked in {elapsed_chunk:.1f}s, embedded in {elapsed_emb:.1f}s")

    print(f"\n{'='*60}")
    print("RUNNING QUERIES")
    print(f"{'='*60}")

    results = []

    for i, case in enumerate(HARD_CASES):
        doc = case["doc"]
        query = case["query"]

        print(f"\n[{i+1}/10] {doc}")
        print(f"  Q: {query}")

        full_text = doc_texts[doc]
        orig_tokens = doc_tokens[doc]
        chunks = doc_chunks[doc]
        headers = doc_headers[doc]
        chunk_embs = doc_embeddings[doc]

        decisions = route(full_text, query, str(SPECS_DIR / f"{doc}.pdf"))
        tk = decisions.get("top_k", 10)
        mt = decisions.get("max_tokens", 512)
        sys_prompt = decisions.get("system_prompt",
            "You are a DoD facilities design expert. Answer based on the provided context only. Be specific.")

        # V2 filter: bi-encoder + header boost + cross-encoder re-rank
        t0 = time.perf_counter()
        top_indices, bi_sims, rerank_scores = rf.filter_precomputed(
            query, top_k=tk, headers=headers, chunk_embs=chunk_embs, chunks=chunks)
        t_filter = time.perf_counter() - t0

        # Requirement extraction on retrieved chunks
        t0 = time.perf_counter()
        raw_ctx = "\n\n".join(chunks[idx] for idx in top_indices)
        raw_tokens = len(raw_ctx.split())
        compressed_ctx = rf.extract_requirements(raw_ctx)
        comp_tok = len(compressed_ctx.split())
        t_extract = time.perf_counter() - t0

        ratio = round(orig_tokens / comp_tok, 1) if comp_tok else 0
        extract_ratio = round(raw_tokens / comp_tok, 1) if comp_tok else 0

        print(f"  Filter: {orig_tokens:,} -> {raw_tokens:,} (retrieval) -> {comp_tok:,} (req extract) = {ratio}x total in {t_filter + t_extract:.1f}s")
        print(f"  Requirement extraction removed {raw_tokens - comp_tok} tokens ({extract_ratio}x within retrieved)")
        print(f"  Top sections:")
        for idx in top_indices[:5]:
            h = headers[idx] if idx < len(headers) else "?"
            rs = rerank_scores.get(idx, 0)
            bs = bi_sims[idx] if idx < len(bi_sims) else 0
            print(f"    [{idx}] {h[:50]:50s} bi={bs:.3f} rerank={rs:.3f}")

        if not wait_for_fatman(120):
            print("  SKIPPING -- Fat Man down")
            results.append({
                "doc": doc, "query": query, "answer": None,
                "error": "Fat Man unreachable",
            })
            continue

        t0 = time.perf_counter()
        answer, usage = chat(
            [{"role": "user", "content": f"Question: {query}\n\nContext:\n{compressed_ctx}"}],
            system=sys_prompt, max_tokens=mt)
        t_answer = time.perf_counter() - t0
        print(f"  Answer: {t_answer:.1f}s, {usage.get('prompt_tokens', '?')} prompt tokens")
        print(f"  Preview: {answer[:150]}...")

        results.append({
            "doc": doc,
            "query": query,
            "answer": answer,
            "orig_tokens": orig_tokens,
            "retrieved_tokens": raw_tokens,
            "comp_tokens": comp_tok,
            "compression_ratio": ratio,
            "extract_ratio": extract_ratio,
            "latency": round(t_answer, 2),
            "filter_time": round(t_filter, 3),
            "extract_time": round(t_extract, 4),
            "prompt_tokens": usage.get("prompt_tokens"),
            "top_sections": [
                {"index": idx, "header": headers[idx] if idx < len(headers) else "",
                 "bi_sim": round(float(bi_sims[idx]), 4) if idx < len(bi_sims) else 0,
                 "rerank_score": round(rerank_scores.get(idx, 0), 4)}
                for idx in top_indices
            ],
            "router": {
                "doc_type": decisions.get("doc_type"),
                "complexity": decisions.get("query_complexity"),
                "top_k": tk,
                "max_tokens": mt,
            },
        })

        if i < len(HARD_CASES) - 1:
            print("  Cooling down 3s...")
            time.sleep(3)

    out_path = OUT_DIR / "answers_v3_quick.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*60}")
    print(f"Saved {len(results)} answers to {out_path}")

    ok = sum(1 for r in results if r.get("answer"))
    fail = sum(1 for r in results if not r.get("answer"))
    print(f"\n{'Doc':<35} {'Total':>6} {'Retr':>6} {'Final':>6} {'ExtR':>5} {'Filter':>7} {'LLM':>7} {'Prompt':>7}")
    print("-" * 85)
    for r in results:
        if r.get("answer"):
            print(f"{r['doc']:<35} {r.get('compression_ratio',0):>5.1f}x "
                  f"{r.get('retrieved_tokens',0):>5} {r.get('comp_tokens',0):>5} "
                  f"{r.get('extract_ratio',0):>4.1f}x "
                  f"{r.get('filter_time',0):>6.2f}s {r.get('latency',0):>6.1f}s "
                  f"{r.get('prompt_tokens','?'):>7}")
    print(f"\nDone: {ok} captured, {fail} failed")


if __name__ == "__main__":
    main()
