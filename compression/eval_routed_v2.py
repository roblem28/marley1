#!/usr/bin/env python3
"""Marley1 Routed Eval v2 — Optimized.

Changes from v1:
- Pre-extracts and pre-chunks all docs at startup (no repeat work)
- Adaptive cooldown: checks Fat Man health, only pauses if needed
- No fixed 10s sleep between queries
"""
import json, sys, os, time, urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pdfplumber
from router import route
from relevance_filter import RelevanceFilter

FAT_MAN_URL = "http://100.97.87.86:8080/v1/chat/completions"
FAT_MAN_HEALTH = "http://100.97.87.86:8080/health"
SPECS_DIR = Path(os.path.expanduser("~/marley1/specs"))
RESULTS_OUT = Path(os.path.expanduser("~/marley1/compression/eval_results_routed_v2.json"))

TEST_CASES = [
    {"doc":"ufc_electrical","query":"What are the electrical design requirements for DoD facilities?"},
    {"doc":"ufc_electrical","query":"What are the requirements for emergency power systems in DoD facilities?"},
    {"doc":"ufc_electrical","query":"What are the grounding and bonding requirements?"},
    {"doc":"ufc_electrical","query":"What are the lighting design standards for DoD facilities?"},
    {"doc":"ufc_electrical","query":"What are the requirements for electrical distribution systems?"},
    {"doc":"ufc_electrical","query":"What are the power quality requirements?"},
    {"doc":"ufc_electrical","query":"What are the requirements for electrical safety and protection systems?"},
    {"doc":"ufc_electrical","query":"What are the metering and monitoring requirements?"},
    {"doc":"ufc_electrical","query":"What are the requirements for renewable energy integration?"},
    {"doc":"ufc_electrical","query":"What are the electrical requirements for exterior and site lighting?"},
    {"doc":"ufc_structural","query":"What are the structural design requirements for DoD buildings?"},
    {"doc":"ufc_structural","query":"What are the seismic design requirements?"},
    {"doc":"ufc_structural","query":"What are the wind load design requirements?"},
    {"doc":"ufc_structural","query":"What are the progressive collapse prevention requirements?"},
    {"doc":"ufc_structural","query":"What are the foundation design requirements?"},
    {"doc":"ufc_structural","query":"What are the steel connection design requirements?"},
    {"doc":"ufc_structural","query":"What are the concrete design requirements?"},
    {"doc":"ufc_structural","query":"What are the load combination requirements?"},
    {"doc":"ufc_structural","query":"What are the blast resistance requirements?"},
    {"doc":"ufc_structural","query":"What are the inspection and quality control requirements for structural work?"},
    {"doc":"ufc_microgrid","query":"What are the design requirements for military microgrids?"},
    {"doc":"ufc_microgrid","query":"What are the energy storage requirements?"},
    {"doc":"ufc_microgrid","query":"How should a microgrid control system be designed?"},
    {"doc":"ufc_microgrid","query":"What are the renewable energy integration requirements for microgrids?"},
    {"doc":"ufc_microgrid","query":"What are the islanding requirements for military microgrids?"},
    {"doc":"ufc_microgrid","query":"What are the cybersecurity requirements for microgrid systems?"},
    {"doc":"ufc_microgrid","query":"What are the generator requirements for military microgrids?"},
    {"doc":"ufc_microgrid","query":"What are the distribution system requirements for microgrids?"},
    {"doc":"ufc_microgrid","query":"What are the testing and commissioning requirements for microgrids?"},
    {"doc":"ufc_microgrid","query":"What are the maintenance and operations requirements for microgrids?"},
    {"doc":"ufc_c5isr_facilities","query":"What are the design requirements for C5ISR facilities?"},
    {"doc":"ufc_c5isr_facilities","query":"What are the telecommunications room requirements?"},
    {"doc":"ufc_c5isr_facilities","query":"What are the equipment room cooling requirements?"},
    {"doc":"ufc_c5isr_facilities","query":"What are the redundancy requirements for C5ISR systems?"},
    {"doc":"ufc_c5isr_facilities","query":"What are the physical security requirements for C5ISR facilities?"},
    {"doc":"ufc_c5isr_facilities","query":"What are the HVAC requirements for equipment rooms?"},
    {"doc":"ufc_c5isr_facilities","query":"What are the power and grounding requirements for C5ISR systems?"},
    {"doc":"ufc_c5isr_facilities","query":"What are the cable plant and infrastructure requirements?"},
    {"doc":"ufc_c5isr_facilities","query":"What are the antenna and RF system requirements?"},
    {"doc":"ufc_c5isr_facilities","query":"What are the testing and acceptance requirements for C5ISR facilities?"},
    {"doc":"ufc_unaccompanied_housing","query":"What are the design standards for military unaccompanied housing?"},
    {"doc":"ufc_unaccompanied_housing","query":"What are the room size and layout requirements?"},
    {"doc":"ufc_unaccompanied_housing","query":"What are the bathroom and toilet facility requirements?"},
    {"doc":"ufc_unaccompanied_housing","query":"What are the common area and amenity requirements?"},
    {"doc":"ufc_unaccompanied_housing","query":"What are the kitchen and dining facility requirements?"},
    {"doc":"ufc_unaccompanied_housing","query":"What are the HVAC and mechanical system requirements?"},
    {"doc":"ufc_unaccompanied_housing","query":"What are the laundry facility requirements?"},
    {"doc":"ufc_unaccompanied_housing","query":"What are the storage and closet space requirements?"},
    {"doc":"ufc_unaccompanied_housing","query":"What are the accessibility and ADA compliance requirements?"},
    {"doc":"ufc_unaccompanied_housing","query":"What are the IT and communications infrastructure requirements?"},
]

JUDGE_PROMPT = """Score the answer below on four criteria, each 0.0 to 1.0.

QUESTION: {query}
ANSWER: {answer}

Return ONLY a JSON object with these exact keys:
factual_accuracy (0.0-1.0): are the facts correct and domain-specific?
completeness (0.0-1.0): are key requirements covered?
coherence (0.0-1.0): is the answer clear and well-structured?
hallucination_risk (0.0-1.0): 1.0 means no hallucination, 0.0 means severe hallucination
verdict: one of equivalent, acceptable, degraded, failed

Respond with only the JSON object, no other text."""


def check_fatman(timeout=5):
    try:
        with urllib.request.urlopen(FAT_MAN_HEALTH, timeout=timeout) as r:
            return r.status == 200
    except:
        return False


def wait_for_fatman(max_wait=120):
    """Adaptive cooldown: only wait if Fat Man is unresponsive."""
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


def judge(query, answer):
    prompt = JUDGE_PROMPT.format(query=query, answer=answer)
    raw, _ = chat([{"role": "user", "content": prompt}],
                  system="You are an objective evaluator. Respond ONLY with valid JSON.", temp=0.0)
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except:
        return {"factual_accuracy": 0, "completeness": 0, "coherence": 0,
                "hallucination_risk": 0, "verdict": "parse_error", "raw": raw[:200]}


def extract_pdf(path):
    pdf = pdfplumber.open(path)
    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    pdf.close()
    return text


def main():
    # Resume support
    log_path = "eval_routed_v2_3s_8k_run.log"
    completed = 0
    try:
        for line in open(log_path):
            if line.strip().startswith("Baseline scores:"):
                completed += 1
    except:
        pass
    skip = completed
    print(f"Skipping {skip} completed cases")

    # =========================================================
    # PRE-LOAD: Extract all docs and chunk them ONCE
    # =========================================================
    print(f"\n{'='*60}")
    print("PRE-LOADING ALL DOCUMENTS")
    print(f"{'='*60}")

    doc_names = sorted(set(c["doc"] for c in TEST_CASES))
    doc_texts = {}
    doc_tokens = {}
    doc_chunks = {}  # keyed by (doc, chunk_size)

    t_preload = time.perf_counter()
    for doc_name in doc_names:
        pdf_path = SPECS_DIR / f"{doc_name}.pdf"
        t0 = time.perf_counter()
        text = extract_pdf(pdf_path)
        elapsed = time.perf_counter() - t0
        tokens = len(text.split())
        doc_texts[doc_name] = text
        doc_tokens[doc_name] = tokens
        print(f"  {doc_name}: {tokens:,} tokens extracted in {elapsed:.1f}s")

    # Pre-chunk with likely chunk sizes
    print("\nPre-chunking documents...")
    rf = RelevanceFilter()
    for doc_name in doc_names:
        for cs in [3, 5, 7]:
            t0 = time.perf_counter()
            chunks = rf.chunk(doc_texts[doc_name], chunk_size=cs)
            elapsed = time.perf_counter() - t0
            doc_chunks[(doc_name, cs)] = chunks
            print(f"  {doc_name} chunk_size={cs}: {len(chunks)} chunks in {elapsed:.1f}s")

    # Pre-compute embeddings for all chunks
    print("\nPre-computing embeddings...")
    doc_embeddings = {}
    for key, chunks in doc_chunks.items():
        t0 = time.perf_counter()
        embeddings = rf.model.encode(chunks, show_progress_bar=False)
        elapsed = time.perf_counter() - t0
        doc_embeddings[key] = embeddings
        print(f"  {key[0]} cs={key[1]}: {len(chunks)} embeddings in {elapsed:.1f}s")

    t_preload_total = time.perf_counter() - t_preload
    print(f"\nPre-load complete: {t_preload_total:.1f}s total")
    print(f"{'='*60}\n")

    # =========================================================
    # EVAL LOOP
    # =========================================================
    results = []

    for i, case in enumerate(TEST_CASES):
        if i < skip:
            continue
        doc = case["doc"]
        query = case["query"]

        print(f"\n{'='*60}")
        print(f"DOC: {doc}")
        print(f"QUERY: {query}")

        full_text = doc_texts[doc]
        orig_tokens = doc_tokens[doc]
        print(f"  Tokens: {orig_tokens:,} (pre-loaded)")

        # Route
        t0 = time.perf_counter()
        decisions = route(full_text, query, str(SPECS_DIR / f"{doc}.pdf"))
        t_route = time.perf_counter() - t0
        dt = decisions.get('doc_type', '?')
        cx = decisions.get('query_complexity', '?')
        cs = decisions.get('chunk_size', 5)
        tk = decisions.get('top_k', 10)
        mt = decisions.get('max_tokens', 512)
        sys_prompt = decisions.get('system_prompt', 'You are a DoD facilities design expert. Answer based on the provided context only. Be specific.')
        print(f"  Router: type={dt} complexity={cx} chunks={cs} topk={tk} max_tokens={mt} ({t_route*1000:.0f}ms)")

        # Relevance filter using pre-computed chunks + embeddings
        t0 = time.perf_counter()
        chunks = doc_chunks[(doc, cs)]
        chunk_embs = doc_embeddings[(doc, cs)]
        query_emb = rf.model.encode([query], show_progress_bar=False)

        import numpy as np
        sims = np.dot(chunk_embs, query_emb.T).flatten()
        top_indices = np.argsort(sims)[-tk:][::-1]
        compressed_ctx = " ".join(chunks[int(idx)] for idx in top_indices)
        t_filter = time.perf_counter() - t0
        comp_tok = len(compressed_ctx.split())
        ratio = round(orig_tokens / comp_tok, 1) if comp_tok else 0
        print(f"  Compressed {orig_tokens:,} -> {comp_tok:,} tokens ({ratio}x) in {t_filter:.1f}s")

        # Adaptive health check before inference
        if not wait_for_fatman(120):
            print("  SKIPPING — Fat Man down")
            continue

        # Compressed answer
        t0 = time.perf_counter()
        comp_answer, comp_usage = chat(
            [{"role": "user", "content": f"Question: {query}\n\nContext:\n{compressed_ctx}"}],
            system=sys_prompt, max_tokens=mt)
        t_comp = time.perf_counter() - t0
        print(f"  Compressed answer: {t_comp:.1f}s, {comp_usage.get('prompt_tokens','?')} prompt tokens")

        # Baseline
        t0 = time.perf_counter()
        base_answer, base_usage = chat(
            [{"role": "user", "content": query}],
            system="You are a DoD facilities design expert. Answer concisely.", max_tokens=512)
        t_base = time.perf_counter() - t0
        print(f"  Baseline answer:   {t_base:.1f}s")

        # Judge
        comp_scores = judge(query, comp_answer)
        base_scores = judge(query, base_answer)
        print(f"  Compressed scores: {comp_scores}")
        print(f"  Baseline scores:   {base_scores}")

        results.append({
            "doc": doc, "query": query,
            "orig_tokens": orig_tokens, "comp_tokens": comp_tok,
            "compression_ratio": ratio,
            "router": {"doc_type": dt, "complexity": cx, "chunk_size": cs,
                       "top_k": tk, "max_tokens": mt},
            "compressed": {"scores": comp_scores, "latency": round(t_comp, 2),
                          "prompt_tokens": comp_usage.get("prompt_tokens")},
            "baseline": {"scores": base_scores, "latency": round(t_base, 2)},
            "filter_time": round(t_filter, 3),
        })
        __import__("time").sleep(3)

    RESULTS_OUT.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_OUT}")

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Doc':<35} {'Ratio':>6} {'Acc C/B':>10} {'Router':>20}")
    print("-" * 90)
    for r in results:
        cs = r["compressed"]["scores"]
        bs = r["baseline"]["scores"]
        ca = cs.get("factual_accuracy", "?")
        ba = bs.get("factual_accuracy", "?")
        rt = r["router"]
        print(f"{r['doc']:<35} {r['compression_ratio']:>5.1f}x  {ca}/{ba}  type={rt['doc_type']} cx={rt['complexity']} tk={rt['top_k']}")

    total = len(results)
    matched = sum(1 for r in results if r["compressed"]["scores"].get("factual_accuracy", 0) >= 0.8)
    failed = sum(1 for r in results if r["compressed"]["scores"].get("factual_accuracy", 1) < 0.5)
    print(f"\nTotal: {total}q, {matched}/{total} match, {failed} fail")


if __name__ == "__main__":
    main()
