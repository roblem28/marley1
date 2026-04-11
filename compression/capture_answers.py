#!/usr/bin/env python3
"""Capture all 50 eval answers from the routed pipeline.

Runs every query through pre-loaded docs + router + relevance filter + Fat Man,
saves the actual answer text (no judging). Outputs:
  - answers_all_50.json  (all 50 in one file)
  - answers/<doc>.json   (per-doc files)
"""
import json, sys, os, time, urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pdfplumber
from router import route
from relevance_filter import RelevanceFilter

FAT_MAN_URL = "http://100.97.87.86:8080/v1/chat/completions"
FAT_MAN_HEALTH = "http://100.97.87.86:8080/health"
SPECS_DIR = Path(os.path.expanduser("~/marley1/specs"))
OUT_DIR = Path(os.path.expanduser("~/marley1/compression"))
ANSWERS_DIR = OUT_DIR / "answers"

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
    ANSWERS_DIR.mkdir(exist_ok=True)

    # =======================================================
    # PRE-LOAD: Extract all docs and chunk them ONCE
    # =======================================================
    print(f"\n{'='*60}")
    print("PRE-LOADING ALL DOCUMENTS")
    print(f"{'='*60}")

    doc_names = sorted(set(c["doc"] for c in TEST_CASES))
    doc_texts = {}
    doc_tokens = {}
    doc_chunks = {}
    doc_embeddings = {}

    t_preload = time.perf_counter()
    for doc_name in doc_names:
        pdf_path = SPECS_DIR / f"{doc_name}.pdf"
        t0 = time.perf_counter()
        text = extract_pdf(pdf_path)
        elapsed = time.perf_counter() - t0
        tokens = len(text.split())
        doc_texts[doc_name] = text
        doc_tokens[doc_name] = tokens
        print(f"  {doc_name}: {tokens:,} tokens in {elapsed:.1f}s")

    print("\nPre-chunking documents...")
    rf = RelevanceFilter()
    for doc_name in doc_names:
        for cs in [3, 5, 7]:
            t0 = time.perf_counter()
            chunks = rf.chunk(doc_texts[doc_name], chunk_size=cs)
            elapsed = time.perf_counter() - t0
            doc_chunks[(doc_name, cs)] = chunks
            print(f"  {doc_name} cs={cs}: {len(chunks)} chunks in {elapsed:.1f}s")

    print("\nPre-computing embeddings...")
    for key, chunks in doc_chunks.items():
        t0 = time.perf_counter()
        embeddings = rf.model.encode(chunks, show_progress_bar=False)
        elapsed = time.perf_counter() - t0
        doc_embeddings[key] = embeddings
        print(f"  {key[0]} cs={key[1]}: {len(chunks)} embeddings in {elapsed:.1f}s")

    t_preload_total = time.perf_counter() - t_preload
    print(f"\nPre-load complete: {t_preload_total:.1f}s total")
    print(f"{'='*60}\n")

    # =======================================================
    # CAPTURE LOOP
    # =======================================================
    all_answers = []
    per_doc = {}

    for i, case in enumerate(TEST_CASES):
        doc = case["doc"]
        query = case["query"]

        print(f"\n[{i+1}/50] {doc}")
        print(f"  Q: {query}")

        full_text = doc_texts[doc]
        orig_tokens = doc_tokens[doc]

        # Route
        t0 = time.perf_counter()
        decisions = route(full_text, query, str(SPECS_DIR / f"{doc}.pdf"))
        t_route = time.perf_counter() - t0
        cs = decisions.get("chunk_size", 5)
        tk = decisions.get("top_k", 10)
        mt = decisions.get("max_tokens", 512)
        sys_prompt = decisions.get("system_prompt",
            "You are a DoD facilities design expert. Answer based on the provided context only. Be specific.")
        print(f"  Router: cs={cs} topk={tk} max_tokens={mt} ({t_route*1000:.0f}ms)")

        # Relevance filter with pre-computed data
        t0 = time.perf_counter()
        chunks = doc_chunks[(doc, cs)]
        chunk_embs = doc_embeddings[(doc, cs)]
        query_emb = rf.model.encode([query], show_progress_bar=False)
        sims = np.dot(chunk_embs, query_emb.T).flatten()
        top_indices = np.argsort(sims)[-tk:][::-1]
        compressed_ctx = " ".join(chunks[int(idx)] for idx in top_indices)
        t_filter = time.perf_counter() - t0
        comp_tok = len(compressed_ctx.split())
        ratio = round(orig_tokens / comp_tok, 1) if comp_tok else 0
        print(f"  Compressed: {orig_tokens:,} -> {comp_tok:,} tokens ({ratio}x) in {t_filter:.1f}s")

        # Health check
        if not wait_for_fatman(120):
            print("  SKIPPING -- Fat Man down")
            entry = {
                "doc": doc, "query": query, "answer": None,
                "error": "Fat Man unreachable",
                "orig_tokens": orig_tokens, "comp_tokens": comp_tok,
                "compression_ratio": ratio,
            }
            all_answers.append(entry)
            per_doc.setdefault(doc, []).append(entry)
            continue

        # Get answer
        t0 = time.perf_counter()
        answer, usage = chat(
            [{"role": "user", "content": f"Question: {query}\n\nContext:\n{compressed_ctx}"}],
            system=sys_prompt, max_tokens=mt)
        t_answer = time.perf_counter() - t0
        print(f"  Answer: {t_answer:.1f}s, {usage.get('prompt_tokens', '?')} prompt tokens")
        print(f"  Preview: {answer[:120]}...")

        entry = {
            "doc": doc,
            "query": query,
            "answer": answer,
            "orig_tokens": orig_tokens,
            "comp_tokens": comp_tok,
            "compression_ratio": ratio,
            "latency": round(t_answer, 2),
            "prompt_tokens": usage.get("prompt_tokens"),
            "router": {
                "doc_type": decisions.get("doc_type"),
                "complexity": decisions.get("query_complexity"),
                "chunk_size": cs,
                "top_k": tk,
                "max_tokens": mt,
            },
        }
        all_answers.append(entry)
        per_doc.setdefault(doc, []).append(entry)

        # 3s cooldown
        if i < len(TEST_CASES) - 1:
            print("  Cooling down 3s...")
            time.sleep(3)

    # =======================================================
    # SAVE
    # =======================================================
    all_path = OUT_DIR / "answers_all_50.json"
    all_path.write_text(json.dumps(all_answers, indent=2))
    print(f"\nSaved {len(all_answers)} answers to {all_path}")

    for doc_name, entries in per_doc.items():
        doc_path = ANSWERS_DIR / f"{doc_name}.json"
        doc_path.write_text(json.dumps(entries, indent=2))
        print(f"  {doc_name}: {len(entries)} answers -> {doc_path}")

    # Summary
    ok = sum(1 for a in all_answers if a.get("answer"))
    fail = sum(1 for a in all_answers if not a.get("answer"))
    print(f"\nDone: {ok} captured, {fail} failed")


if __name__ == "__main__":
    main()
