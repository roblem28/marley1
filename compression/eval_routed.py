#!/usr/bin/env python3
"""Marley1 Routed Eval Framework.

Same 50 queries as eval_framework.py but uses the symbolic router
for adaptive chunk_size, top_k, max_tokens, and system prompts.

Old eval files are untouched.
"""
import json, sys, os, time, urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pdfplumber
from router import route
from relevance_filter import RelevanceFilter
from abbrev import ConstructionCompressor

FAT_MAN_URL = "http://100.97.87.86:8080/v1/chat/completions"
SPECS_DIR = Path(os.path.expanduser("~/marley1/specs"))
RESULTS_OUT = Path(os.path.expanduser("~/marley1/compression/eval_results_routed.json"))

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
                time.sleep(5)
                print(f"  Retry {attempt+1}/2: {e}")
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
    completed = set()
    log_path = "eval_routed_run.log"
    try:
        for line in open(log_path):
            if line.strip().startswith("Baseline scores:"):
                completed.add(len(completed))
    except:
        pass
    skip = len(completed)
    print(f"Skipping {skip} completed cases")

    rf = RelevanceFilter()
    results = []

    for i, case in enumerate(TEST_CASES):
        if i < skip:
            continue
        doc = case["doc"]
        query = case["query"]
        pdf_path = SPECS_DIR / f"{doc}.pdf"

        print(f"\n{'='*60}")
        print(f"DOC: {doc}")
        print(f"QUERY: {query}")

        # Extract
        full_text = extract_pdf(pdf_path)
        orig_tokens = len(full_text.split())
        print(f"  Extracted ~{orig_tokens:,} tokens")

        # Route (symbolic)
        t0 = time.perf_counter()
        decisions = route(full_text, query, str(pdf_path))
        t_route = time.perf_counter() - t0
        dt = decisions.get('doc_type', '?')
        cx = decisions.get('query_complexity', '?')
        cs = decisions.get('chunk_size', 5)
        tk = decisions.get('top_k', 10)
        mt = decisions.get('max_tokens', 512)
        sys_prompt = decisions.get('system_prompt', 'You are a DoD facilities design expert. Answer based on the provided context only. Be specific.')
        print(f"  Router: type={dt} complexity={cx} chunks={cs} topk={tk} max_tokens={mt} ({t_route*1000:.0f}ms)")

        # Compress with routed params
        t0 = time.perf_counter()
        chunks = rf.chunk(full_text, chunk_size=cs)
        kept_indices, similarities = rf.filter(chunks, query, top_k=tk)
        compressed_ctx = " ".join(chunks[int(idx)] for idx in kept_indices)
        t_compress = time.perf_counter() - t0
        comp_tok = len(compressed_ctx.split())
        ratio = round(orig_tokens / comp_tok, 1) if comp_tok else 0
        print(f"  Compressed {orig_tokens:,} -> {comp_tok:,} tokens ({ratio}x) in {t_compress:.1f}s")

        # Compressed answer with routed system prompt and max_tokens
        t0 = time.perf_counter()
        comp_answer, comp_usage = chat(
            [{"role": "user", "content": f"Question: {query}\n\nContext:\n{compressed_ctx}"}],
            system=sys_prompt, max_tokens=mt)
        t_comp = time.perf_counter() - t0
        print(f"  Compressed answer: {t_comp:.1f}s, {comp_usage.get('prompt_tokens','?')} prompt tokens")

        # Baseline (no context, generic prompt)
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
        })

        # Cooldown
        __import__("time").sleep(10)

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
