#!/usr/bin/env python3
import json, sys, os, time, re, glob, urllib.request
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/marley1/compression"))
import numpy as np
from router import route, classify_answer_mode, expand_query, MODE_MAX_TOKENS
from relevance_filter_v2 import (RelevanceFilterV4, extract_text_no_toc, extract_pdf_tables, _TOC_LINE_RE, detect_family)

OSHA_DIR = os.path.expanduser("~/marley1/benchmarks/osha")
SUBPART_DIR = os.path.join(OSHA_DIR, "subparts")
PDF_PATH = os.path.join(OSHA_DIR, "OSHA_1926_only.pdf")
FM_HEALTH = "http://100.97.87.86:8080/health"
FM_URL = "http://100.97.87.86:8080/v1/chat/completions"

def find_subpart_file(letter):
    pattern = os.path.join(SUBPART_DIR, "subpart_" + letter + "_*.txt")
    files = glob.glob(pattern)
    if not files: return None
    return max(files, key=os.path.getsize)

def check_fm(t=5):
    try:
        with urllib.request.urlopen(FM_HEALTH, timeout=t) as r: return r.status==200
    except: return False

def chat(msgs, system=None, mt=160, temp=0.1):
    body = {"model":"qwen2.5-3b","messages":msgs,"max_tokens":mt,"temperature":temp}
    if system: body["messages"] = [{"role":"system","content":system}]+msgs
    data = json.dumps(body).encode()
    req = urllib.request.Request(FM_URL, data, {"Content-Type":"application/json"})
    for a in range(2):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"], True
        except:
            if a==0: time.sleep(3)
    return None, False

def run_osha_query(rf, text, query, pdf_path=None, pdf_tables=None):
    decisions = route(text, query, pdf_path)
    am=decisions["answer_mode"]; eq=decisions["expanded_query"]
    cap=decisions.get("cap_expansion",False); mt=decisions["max_tokens"]
    sp=decisions.get("system_prompt",""); tk=decisions["top_k"]
    family = detect_family(query)
    chunks, headers = rf.chunk_with_headers(text, pdf_tables=pdf_tables or [])
    if not chunks:
        return {"final_answer":"No chunks.","used_deterministic":False,"used_llm":False,"proof_quality_score":0,"answer_mode":am,"orig_tokens":len(text.split()),"evidence_tokens":0,"compression_ratio":0,"truncation_occurred":False,"evidence_dropped":False,"top_sections":[],"chosen_evidence_text":[],"final_answerability":{},"deterministic_stop_reason":"no_chunks"}
    embs, bm25 = rf.precompute(chunks)
    ti, ds, ss, rr = rf.filter_two_lane(query, eq, top_k=tk, headers=headers, chunk_embs=embs, chunks=chunks, bm25_index=bm25, answer_mode=am, cap_expansion=cap)
    NL = chr(10)
    raw_ctx = (NL+NL).join(chunks[j] for j in ti)
    evidence = rf.extract_evidence(raw_ctx)
    et = len(evidence.split()) if evidence else 0
    ot = len(text.split()); ratio = round(ot/et, 1) if et else 0
    ans_score = rf.score_answerability(evidence, query)
    answer=None; used_det=False; used_llm=False; det_stop=None; fallback=None
    det_ans, det_used, det_stop, fam_applied, clipped = rf.deterministic_answer(am, evidence, query, family=family, headers=headers, chunks=chunks, top_indices=ti)
    if det_used and det_ans: answer=det_ans; used_det=True
    if not answer and ans_score.get("coverage") != "unsupported":
        if check_fm():
            used_llm = True
            prompt = "Question: " + query + NL + NL + "Evidence:" + NL + evidence
            raw, ok = chat([{"role":"user","content":prompt}], system=sp, mt=mt)
            if ok and raw: answer = rf.verify_answer(raw, evidence)
    if not answer and evidence and evidence.strip():
        fb_ans, fb_mode = rf.extractive_fallback(evidence, query, am, family, headers, chunks, ti)
        if fb_ans: answer=fb_ans; fallback=fb_mode
    if not answer: answer = "Not specified in the retrieved sections."
    pq = rf.proof_quality_score(evidence, answer, family)
    ev_lines = [l for l in (evidence or "").strip().split(NL) if l.strip()]
    return {"final_answer":answer,"answer_mode":am,"used_deterministic":used_det,"used_llm":used_llm,"fallback_mode_used":fallback,"final_answerability":ans_score,"proof_quality_score":pq,"deterministic_stop_reason":det_stop,"chosen_evidence_text":ev_lines[:10],"orig_tokens":ot,"evidence_tokens":et,"compression_ratio":ratio,"truncation_occurred":et>2000,"evidence_dropped":False,"top_sections":[{"index":j,"header":headers[j] if j<len(headers) else "","rerank":round(rr.get(j,0),4)} for j in ti[:5]]}

def classify_shape(answer):
    if not answer: return "empty"
    if "Not specified" in answer: return "abstention"
    NL = chr(10)
    lines = [l for l in answer.strip().split(NL) if l.strip()]
    if len(lines)==1 and len(answer.split())<20: return "one_governing_line"
    if len(lines)>=2: return "behavior_bullets"
    return "one_sentence"

def main():
    sep = "=" * 70
    print(sep); print("OSHA 1926 TRANSFER v1 -- v4.6 unchanged"); print(sep)
    questions = json.loads(Path(os.path.join(OSHA_DIR, "osha_1926_eval_15.json")).read_text())
    print("Loaded %d questions" % len(questions))
    print("Loading models..."); rf = RelevanceFilterV4()
    # MERGED
    print(sep); print("MODE A: MERGED CORPUS"); print(sep)
    merged_text, toc_skip = extract_text_no_toc(PDF_PATH)
    merged_tables = extract_pdf_tables(PDF_PATH)
    print("  %d words, %d tables" % (len(merged_text.split()), len(merged_tables)))
    merged_results = []
    for i, q in enumerate(questions):
        query=q.get("question",q.get("query",""))
        subpart=q.get("subpart",""); family=q.get("type",q.get("family",""))
        print("  [%d/15] [%s] %s" % (i+1, family, query[:60]))
        r = run_osha_query(rf, merged_text, query, PDF_PATH, merged_tables)
        r.update({"doc":"merged","query":query,"subpart":subpart,"family":family,"expected_sections":q.get("expected_sections",[]),"actual_answer_shape":classify_shape(r["final_answer"])})
        merged_results.append(r)
        d="Y" if r["used_deterministic"] else "N"; l="Y" if r["used_llm"] else "N"
        print("    PQ=%d det=%s llm=%s" % (r["proof_quality_score"], d, l))
        print("    " + r["final_answer"][:100])
        if i<14: time.sleep(5)
    Path(os.path.join(OSHA_DIR,"answers_osha_merged_v1.json")).write_text(json.dumps(merged_results,indent=2))
    print("Saved merged (%d)" % len(merged_results))
    # SPLIT
    print(sep); print("MODE B: SUBPART-SPLIT"); print(sep)
    split_results = []
    for i, q in enumerate(questions):
        query=q.get("question",q.get("query",""))
        subpart=q.get("subpart",""); family=q.get("type",q.get("family",""))
        print("  [%d/15] [%s] Sub %s: %s" % (i+1, family, subpart, query[:55]))
        sf = find_subpart_file(subpart)
        if not sf:
            print("    SKIP: no file for subpart %s" % subpart)
            r = {"final_answer":"SKIP: subpart not found","answer_mode":"skip","used_deterministic":False,"used_llm":False,"proof_quality_score":0,"final_answerability":{},"chosen_evidence_text":[],"orig_tokens":0,"evidence_tokens":0,"compression_ratio":0,"truncation_occurred":False,"evidence_dropped":False,"top_sections":[],"deterministic_stop_reason":"no_file"}
        else:
            sub_text = open(sf).read()
            print("    File: %s (%d words)" % (os.path.basename(sf), len(sub_text.split())))
            r = run_osha_query(rf, sub_text, query, pdf_tables=[])
        r.update({"doc":"subpart_"+subpart,"query":query,"subpart":subpart,"family":family,"expected_sections":q.get("expected_sections",[]),"actual_answer_shape":classify_shape(r["final_answer"])})
        split_results.append(r)
        d="Y" if r["used_deterministic"] else "N"; l="Y" if r["used_llm"] else "N"
        print("    PQ=%d det=%s llm=%s" % (r["proof_quality_score"], d, l))
        print("    " + r["final_answer"][:100])
        if i<14: time.sleep(5)
    Path(os.path.join(OSHA_DIR,"answers_osha_split_v1.json")).write_text(json.dumps(split_results,indent=2))
    print("Saved split (%d)" % len(split_results))
    # SUMMARY
    print(sep); print("SUMMARY"); print(sep)
    print("%-3s %-20s %-4s %5s %5s %3s %3s" % ("#","Family","Sub","M-PQ","S-PQ","M","S"))
    print("-"*55)
    for i,(m,s) in enumerate(zip(merged_results,split_results)):
        md="D" if m["used_deterministic"] else ("L" if m["used_llm"] else "A")
        sd="D" if s["used_deterministic"] else ("L" if s["used_llm"] else "A")
        print("%-3d %-20s %-4s %5d %5d %3s %3s" % (i+1,m["family"][:18],m["subpart"],m["proof_quality_score"],s["proof_quality_score"],md,sd))
    ma=sum(r["proof_quality_score"] for r in merged_results)/15.0
    sa=sum(r["proof_quality_score"] for r in split_results)/15.0
    mab=sum(1 for r in merged_results if "Not specified" in r["final_answer"] or "SKIP" in r["final_answer"])
    sab=sum(1 for r in split_results if "Not specified" in r["final_answer"] or "SKIP" in r["final_answer"])
    mdet=sum(1 for r in merged_results if r["used_deterministic"])
    sdet=sum(1 for r in split_results if r["used_deterministic"])
    mllm=sum(1 for r in merged_results if r["used_llm"])
    sllm=sum(1 for r in split_results if r["used_llm"])
    print("Merged: avg_pq=%.1f det=%d llm=%d abstain=%d" % (ma,mdet,mllm,mab))
    print("Split:  avg_pq=%.1f det=%d llm=%d abstain=%d" % (sa,sdet,sllm,sab))
    better = "MERGED" if ma>sa else ("SPLIT" if sa>ma else "TIE")
    print("Better: %s" % better)

if __name__ == "__main__": main()
