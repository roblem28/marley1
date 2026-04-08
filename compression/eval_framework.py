#!/usr/bin/env python3
"""
eval_framework.py — Marley1 Quality Evaluator v2
Compressed vs baseline comparison with LLM-as-judge scoring.
"""
import json, time, urllib.request, sys, os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdfplumber
from relevance_filter import RelevanceFilter
from abbrev import ConstructionCompressor

FAT_MAN_URL = "http://100.97.87.86:8080/v1/chat/completions"
SPECS_DIR   = Path("~/marley1/specs").expanduser()
RESULTS_OUT = Path("~/marley1/compression/eval_results_v2.json").expanduser()

TEST_CASES = [
    {"doc": "ufc_electrical", "query": "What are the electrical design requirements for DoD facilities?"},
    {"doc": "ufc_electrical", "query": "What are the requirements for emergency power systems in DoD facilities?"},
    {"doc": "ufc_electrical", "query": "What are the grounding and bonding requirements?"},
    {"doc": "ufc_electrical", "query": "What are the lighting design standards for DoD facilities?"},
    {"doc": "ufc_electrical", "query": "What are the requirements for electrical distribution systems?"},
    {"doc": "ufc_electrical", "query": "What are the power quality requirements?"},
    {"doc": "ufc_electrical", "query": "What are the requirements for electrical safety and protection systems?"},
    {"doc": "ufc_electrical", "query": "What are the metering and monitoring requirements?"},
    {"doc": "ufc_electrical", "query": "What are the requirements for renewable energy integration?"},
    {"doc": "ufc_electrical", "query": "What are the electrical requirements for exterior and site lighting?"},
    {"doc": "ufc_structural", "query": "What are the structural design requirements for DoD buildings?"},
    {"doc": "ufc_structural", "query": "What are the seismic design requirements?"},
    {"doc": "ufc_structural", "query": "What are the wind load design requirements?"},
    {"doc": "ufc_structural", "query": "What are the progressive collapse prevention requirements?"},
    {"doc": "ufc_structural", "query": "What are the foundation design requirements?"},
    {"doc": "ufc_structural", "query": "What are the requirements for structural steel design?"},
    {"doc": "ufc_structural", "query": "What are the concrete design and construction requirements?"},
    {"doc": "ufc_structural", "query": "What are the blast resistance requirements?"},
    {"doc": "ufc_structural", "query": "What are the requirements for roof structural systems?"},
    {"doc": "ufc_structural", "query": "What are the inspection and quality control requirements for structural work?"},
    {"doc": "ufc_microgrid", "query": "What are the design requirements for installation microgrids?"},
    {"doc": "ufc_microgrid", "query": "What are the islanding and grid disconnect requirements?"},
    {"doc": "ufc_microgrid", "query": "What are the energy storage requirements for microgrids?"},
    {"doc": "ufc_microgrid", "query": "What are the control and automation requirements for microgrids?"},
    {"doc": "ufc_microgrid", "query": "What are the requirements for distributed generation in microgrids?"},
    {"doc": "ufc_microgrid", "query": "What are the cybersecurity requirements for microgrid systems?"},
    {"doc": "ufc_microgrid", "query": "What are the testing and commissioning requirements for microgrids?"},
    {"doc": "ufc_microgrid", "query": "What are the resilience and reliability requirements?"},
    {"doc": "ufc_microgrid", "query": "What are the interconnection requirements with the utility grid?"},
    {"doc": "ufc_microgrid", "query": "What are the operations and maintenance requirements for microgrids?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the requirements for C5ISR facility design?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the electromagnetic interference shielding requirements?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the TEMPEST requirements for C5ISR facilities?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the physical security requirements for C5ISR facilities?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the HVAC requirements for equipment rooms?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the power and grounding requirements for C5ISR systems?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the cable plant and infrastructure requirements?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the antenna and RF system requirements?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the requirements for operations centers within C5ISR facilities?"},
    {"doc": "ufc_c5isr_facilities", "query": "What are the testing and acceptance requirements for C5ISR facilities?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the design standards for military unaccompanied housing?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the room size and layout requirements?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the bathroom and toilet facility requirements?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the common area and amenity requirements?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the kitchen and dining facility requirements?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the HVAC and mechanical system requirements?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the laundry facility requirements?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the storage and closet space requirements?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the accessibility and ADA compliance requirements?"},
    {"doc": "ufc_unaccompanied_housing", "query": "What are the IT and communications infrastructure requirements?"},
]

JUDGE_PROMPT = """You are a strict evaluator. Score the answer below on four criteria, each 0.0 to 1.0.

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
    body = {"model":"qwen2.5-3b-instruct-q4_k_m.gguf","messages":messages,"max_tokens":max_tokens,"temperature":temp}
    if system:
        body["messages"] = [{"role":"system","content":system}] + messages
    req = urllib.request.Request(FAT_MAN_URL, json.dumps(body).encode(), {"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        obj = json.loads(r.read())
    return obj["choices"][0]["message"]["content"], obj.get("usage",{})

def judge(query, answer):
    prompt = JUDGE_PROMPT.format(query=query, answer=answer)
    raw, _ = chat([{"role":"user","content":prompt}],
                  system="You are an objective evaluator. Respond ONLY with valid JSON.", temp=0.0)
    try:
        s = raw.find("{"); e = raw.rfind("}")+1
        return json.loads(raw[s:e])
    except:
        return {"error":"parse_failed","raw":raw[:200]}

def extract_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)

def compress(full_text, query, rf, abbrev, top_k=5):
    chunks = rf.chunk(full_text, chunk_size=20)
    kept, sims = rf.filter(chunks, query, top_k=top_k)
    filtered = " ".join(chunks[int(i)] for i in kept)
    compressed = abbrev.compress(filtered)
    return compressed, len(full_text.split()), len(compressed.split())

def main():
    results = []
    rf = RelevanceFilter()
    abbrev = ConstructionCompressor()

    completed = set()
    try:
        for line in open('eval_run.log'):
            if line.startswith('DOC: '):
                d = line.strip().split('DOC: ')[1]
            if line.strip().startswith('Baseline scores:'):
                completed.add((d, len(completed)))
    except: pass
    skip = len(completed)
    print(f'Skipping {skip} completed cases')
    for i, case in enumerate(TEST_CASES):
        if i < skip:
            continue
        doc   = case["doc"]
        query = case["query"]
        pdf_path = SPECS_DIR / f"{doc}.pdf"

        print(f"\n{'='*60}")
        print(f"DOC: {doc}")
        print(f"QUERY: {query}")

        full_text = extract_pdf(pdf_path)
        orig_tokens = len(full_text.split())
        print(f"  Extracted ~{orig_tokens:,} tokens")

        t0 = time.perf_counter()
        compressed_ctx, orig_tok, comp_tok = compress(full_text, query, rf, abbrev)
        t_compress = time.perf_counter() - t0
        ratio = round(orig_tok / comp_tok, 1) if comp_tok else 0
        print(f"  Compressed {orig_tok:,} -> {comp_tok:,} tokens ({ratio}x) in {t_compress:.1f}s")

        system = "You are a DoD facilities design expert. Answer based on the provided context only. Be specific."

        t0 = time.perf_counter()
        comp_answer, comp_usage = chat(
            [{"role":"user","content":f"Question: {query}\n\nContext:\n{compressed_ctx}"}],
            system=system, max_tokens=512)
        t_comp = time.perf_counter() - t0
        print(f"  Compressed answer: {t_comp:.1f}s, {comp_usage.get('prompt_tokens','?')} prompt tokens")

        t0 = time.perf_counter()
        base_answer, base_usage = chat(
            [{"role":"user","content":query}],
            system="You are a DoD facilities design expert. Answer concisely.", max_tokens=512)
        t_base = time.perf_counter() - t0
        print(f"  Baseline answer:   {t_base:.1f}s")

        comp_scores = judge(query, comp_answer)
        base_scores = judge(query, base_answer)
        print(f"  Compressed scores: {comp_scores}")
        print(f"  Baseline scores:   {base_scores}")

        results.append({
            "doc": doc,
            "query": query,
            "orig_tokens": orig_tok,
            "comp_tokens": comp_tok,
            "compression_ratio": ratio,
            "compressed": {"answer": comp_answer, "scores": comp_scores, "latency": round(t_comp,2), "prompt_tokens": comp_usage.get("prompt_tokens")},
            "baseline":   {"answer": base_answer, "scores": base_scores, "latency": round(t_base,2)},
        })
        __import__("time").sleep(10)

    RESULTS_OUT.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_OUT}")

    print(f"\n{'='*90}")
    print(f"{'Doc':<30} {'Ratio':>6}  {'Acc C/B':>8}  {'Comp C/B':>9}  {'Halluc C/B':>11}  {'Verdict C/B'}")
    print("-"*90)
    for r in results:
        cs = r["compressed"]["scores"]
        bs = r["baseline"]["scores"]
        if "error" not in cs and "error" not in bs:
            print(f"{r['doc']:<30} {r['compression_ratio']:>5}x"
                  f"  {cs.get('factual_accuracy',0):.2f}/{bs.get('factual_accuracy',0):.2f}"
                  f"  {cs.get('completeness',0):.2f}/{bs.get('completeness',0):.2f}"
                  f"  {cs.get('hallucination_risk',0):.2f}/{bs.get('hallucination_risk',0):.2f}"
                  f"  {cs.get('verdict','?')} / {bs.get('verdict','?')}")

if __name__ == "__main__":
    main()
