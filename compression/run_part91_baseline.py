#!/usr/bin/env python3
"""Part 91 transfer baseline runner."""
import json, sys, os, time, re, urllib.request
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/marley1/compression"))
import numpy as np
from router import route
from relevance_filter_v2 import RelevanceFilterV4, detect_family

P91_DIR = os.path.expanduser("~/marley1/benchmarks/part91")
TXT_PATH = os.path.join(P91_DIR, "14_CFR_Part_91.txt")
EVAL_PATH = os.path.join(P91_DIR, "part91_eval_15.json")
FM_HEALTH = "http://100.97.87.86:8080/health"
FM_URL = "http://100.97.87.86:8080/v1/chat/completions"

def check_fm(t=5):
    try:
        with urllib.request.urlopen(FM_HEALTH, timeout=t) as r: return r.status==200
    except: return False

def chat(msgs, system=None, mt=160):
    body = {"model":"qwen2.5-3b","messages":msgs,"max_tokens":mt,"temperature":0.1}
    if system: body["messages"] = [{"role":"system","content":system}]+msgs
    data = json.dumps(body).encode()
    req = urllib.request.Request(FM_URL, data, {"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"], True
    except: return None, False

def classify_shape(answer):
    if not answer: return "empty"
    if "Not specified" in answer or "insufficient" in answer.lower(): return "abstention"
    lines = [l for l in answer.strip().split("\n") if l.strip()]
    if len(lines) == 1 and len(answer.split()) < 25: return "one_governing_line"
    if any(re.search(r"\d+\s*(?:feet|ft|knots|inches|hours|minutes|MSL|AGL)", l, re.I) for l in lines):
        return "numeric_threshold"
    if any(re.search(r"\bexcept|unless|does\s+not\s+apply", l, re.I) for l in lines):
        return "exception_clause"
    if any(re.search(r"\bwhen|before|after|whenever|prior\s+to", l, re.I) for l in lines):
        return "condition_action"
    if len(lines) >= 2: return "requirement_list"
    return "one_sentence"

def run_query(rf, text, query, expected_sections):
    decisions = route(text, query, None)
    am = decisions["answer_mode"]
    eq = decisions["expanded_query"]
    cap = decisions.get("cap_expansion", False)
    mt = decisions["max_tokens"]
    sp = decisions.get("system_prompt", "")
    tk = decisions["top_k"]
    family = detect_family(query)
    chunks, headers = rf.chunk_with_headers(text, pdf_tables=[])
    if not chunks:
        return {"final_answer":"No chunks.","used_deterministic":False,"used_llm":False,
                "proof_quality_score":0,"answer_mode":am,"orig_tokens":len(text.split()),
                "evidence_tokens":0,"compression_ratio":0,"truncation_occurred":False,
                "evidence_dropped":False,"top_sections":[],"chosen_evidence_text":[],
                "final_answerability":{},"deterministic_stop_reason":"no_chunks",
                "router_decision":decisions,"family_detection":family,
                "expected_section_in_top5":False,"expected_section_in_evidence":False,
                "actual_answer_shape":"empty"}
    embs, bm25 = rf.precompute(chunks)
    ti, ds, ss, rr = rf.filter_two_lane(query, eq, top_k=tk, headers=headers,
        chunk_embs=embs, chunks=chunks, bm25_index=bm25, answer_mode=am, cap_expansion=cap)
    raw_ctx = "\n\n".join(chunks[j] for j in ti)
    evidence = rf.extract_evidence(raw_ctx)
    et = len(evidence.split()) if evidence else 0
    ot = len(text.split())
    ratio = round(ot/et, 1) if et else 0
    ans_score = rf.score_answerability(evidence, query)
    answer = None; used_det = False; used_llm = False; det_stop = None; fallback = None
    det_ans, det_used, det_stop, fam_applied, _ = rf.deterministic_answer(
        am, evidence, query, family=family, headers=headers, chunks=chunks, top_indices=ti)
    if det_used and det_ans: answer = det_ans; used_det = True
    if not answer and ans_score.get("coverage") != "unsupported":
        if check_fm():
            used_llm = True
            raw, ok = chat([{"role":"user","content":"Question: "+query+"\n\nEvidence:\n"+evidence}], system=sp, mt=mt)
            if ok and raw: answer = rf.verify_answer(raw, evidence)
    if not answer and evidence and evidence.strip():
        fb_ans, fb_mode = rf.extractive_fallback(evidence, query, am, family, headers, chunks, ti)
        if fb_ans: answer = fb_ans; fallback = fb_mode
    if not answer: answer = "Not specified in the retrieved sections."
    pq = rf.proof_quality_score(evidence, answer, family)
    ev_lines = [l for l in (evidence or "").strip().split("\n") if l.strip()]
    top_section_hit = False
    chosen_evidence_hit = False
    for exp in expected_sections:
        for j in ti[:5]:
            if j < len(headers) and exp in headers[j]:
                top_section_hit = True; break
        for ev_line in ev_lines:
            if exp in ev_line:
                chosen_evidence_hit = True; break
    return {
        "final_answer": answer, "answer_mode": am, "used_deterministic": used_det,
        "used_llm": used_llm, "fallback_mode_used": fallback, "final_answerability": ans_score,
        "proof_quality_score": pq, "deterministic_stop_reason": det_stop,
        "chosen_evidence_text": ev_lines[:10], "orig_tokens": ot, "evidence_tokens": et,
        "compression_ratio": ratio, "truncation_occurred": et > 2000, "evidence_dropped": False,
        "top_sections": [{"index": j, "header": headers[j] if j < len(headers) else "",
            "rerank": round(rr.get(j, 0), 4)} for j in ti[:8]],
        "router_decision": {"answer_mode": am, "doc_type": decisions.get("doc_type"), "top_k": tk, "max_tokens": mt},
        "family_detection": family,
        "expected_section_in_top5": top_section_hit,
        "expected_section_in_evidence": chosen_evidence_hit,
        "actual_answer_shape": classify_shape(answer),
    }

def main():
    sep = "=" * 70
    print(sep + "\nPART 91 TRANSFER BASELINE -- post-OSHA v2 pipeline\n" + sep)
    questions = json.loads(open(EVAL_PATH).read())
    print("Loaded %d questions" % len(questions))
    print("Loading text corpus...")
    text = open(TXT_PATH).read()
    print("  %d words" % len(text.split()))
    print("Loading models...")
    rf = RelevanceFilterV4()
    results = []
    for i, q in enumerate(questions):
        query = q["query"]
        family_label = q.get("family", "")
        expected = q.get("expected_sections", [])
        print("\n  [%d/15] [%s] %s..." % (i+1, family_label, query[:55]))
        r = run_query(rf, text, query, expected)
        r.update({"id": q["id"], "query": query, "family": family_label,
                  "expected_sections": expected,
                  "expected_answer_shape": q.get("expected_answer_shape", ""),
                  "notes": q.get("notes", "")})
        results.append(r)
        d = "Y" if r["used_deterministic"] else "N"
        l = "Y" if r["used_llm"] else "N"
        print("    PQ=%d det=%s llm=%s top5_hit=%s ev_hit=%s" % (r["proof_quality_score"], d, l, r["expected_section_in_top5"], r["expected_section_in_evidence"]))
        print("    " + r["final_answer"][:100])
        if i < 14: time.sleep(5)
    out = os.path.join(P91_DIR, "answers_part91_baseline.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to " + out)

if __name__ == "__main__": main()
