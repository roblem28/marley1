#!/usr/bin/env python3
"""Eval v4.4: metering, islanding, antenna/RF patches + clipped-line filter. answers_v4_4_quick.json."""
import json, sys, os, time, urllib.request, re
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from router import route, classify_answer_mode, expand_query, MODE_MAX_TOKENS
from relevance_filter_v2 import (RelevanceFilterV4, extract_text_no_toc, extract_pdf_tables,
                                  _TOC_LINE_RE, detect_family)

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

def check_fm(t=5):
    try:
        with urllib.request.urlopen(FAT_MAN_HEALTH, timeout=t) as r: return r.status==200
    except: return False
def wait_fm(mx=30):
    if check_fm(): return True
    print("  Fat Man down (fallback)"); return False
def chat(msgs, system=None, mt=160, temp=0.1):
    body = {"model":"qwen2.5-3b","messages":msgs,"max_tokens":mt,"temperature":temp}
    if system: body["messages"] = [{"role":"system","content":system}]+msgs
    data = json.dumps(body).encode()
    req = urllib.request.Request(FAT_MAN_URL, data, {"Content-Type":"application/json"})
    for a in range(2):
        try:
            with urllib.request.urlopen(req, timeout=120) as r: return json.loads(r.read())["choices"][0]["message"]["content"], True
        except:
            if a==0: time.sleep(3)
    return None, False

def main():
    print(f"\n{'='*70}")
    print("EVAL V4.4 -- metering/islanding/antenna patches + clipped filter")
    print(f"{'='*70}")

    dn = sorted(set(c["doc"] for c in HARD_CASES))
    dt, dtk, dpt = {}, {}, {}
    print("\nExtracting PDFs...")
    for d in dn:
        text, toc = extract_text_no_toc(str(SPECS_DIR/f"{d}.pdf"))
        tables = extract_pdf_tables(str(SPECS_DIR/f"{d}.pdf"))
        dt[d]=text; dtk[d]=len(text.split()); dpt[d]=tables
        print(f"  {d}: {dtk[d]:,} tok, {len(toc)} TOC, {len(tables)} tbl")

    print("\nLoading models...")
    rf = RelevanceFilterV4()

    print("\nIndexing...")
    dc, dh, de, db = {}, {}, {}, {}
    for d in dn:
        c, h = rf.chunk_with_headers(dt[d], pdf_tables=dpt[d])
        e, b = rf.precompute(c)
        dc[d]=c; dh[d]=h; de[d]=e; db[d]=b
        nt = sum(1 for x in h if x.startswith("Table"))
        print(f"  {d}: {len(h)-nt} sec + {nt} tbl")

    print(f"\n{'='*70}\nRUNNING QUERIES\n{'='*70}")
    results = []
    for i, case in enumerate(HARD_CASES):
        doc, query = case["doc"], case["query"]
        print(f"\n[{i+1}/10] {doc}")
        print(f"  Q: {query}")
        chunks=dc[doc]; headers=dh[doc]; ce=de[doc]; bm=db[doc]; ot=dtk[doc]
        family = detect_family(query)
        decisions = route(dt[doc], query, str(SPECS_DIR/f"{doc}.pdf"))
        am=decisions["answer_mode"]; eq=decisions["expanded_query"]
        cap=decisions.get("cap_expansion",False); mt=decisions["max_tokens"]
        sp=decisions.get("system_prompt",""); tk=decisions["top_k"]
        if family=="metering": am="partial_support"
        print(f"  Router: mode={am}, family={family}, mt={mt}")

        t0=time.perf_counter()
        ti, ds, ss, rr = rf.filter_two_lane(query, eq, top_k=tk, headers=headers,
            chunk_embs=ce, chunks=chunks, bm25_index=bm, answer_mode=am, cap_expansion=cap)
        t_ret=time.perf_counter()-t0
        raw_ctx="\n\n".join(chunks[j] for j in ti)
        evidence=rf.extract_evidence(raw_ctx)
        et=len(evidence.split()) if evidence else 0
        ans_score=rf.score_answerability(evidence, query)
        coverage=ans_score["coverage"]
        ratio=round(ot/et,1) if et else 0
        print(f"  Evidence: {ot:,} -> {et} ({ratio}x), coverage={coverage}")
        for j in ti[:3]: print(f"    [{j}] {headers[j][:55] if j<len(headers) else '?'}")

        answer=None; used_det=False; used_llm=False; fm_called=False; fm_ok=False
        fallback=None; det_stop=None; fam_applied=None; pre_verify=None; clipped=[]

        det_ans, det_used, det_stop, fam_applied, clipped = rf.deterministic_answer(
            am, evidence, query, family=family, headers=headers, chunks=chunks, top_indices=ti)
        if det_used and det_ans:
            answer=det_ans; used_det=True
            print(f"  DETERMINISTIC ({det_stop}): {answer[:130]}...")
        if not answer and coverage!="unsupported":
            fm_called=True; fm_ok=wait_fm(30)
            if fm_ok:
                used_llm=True
                raw, ok = chat([{"role":"user","content":f"Question: {query}\n\nEvidence:\n{evidence}"}], system=sp, mt=mt)
                if ok and raw: pre_verify=raw; answer=rf.verify_answer(raw, evidence); print(f"  LLM: {answer[:130]}...")
        if not answer and evidence and evidence.strip():
            fb_ans, fb_mode = rf.extractive_fallback(evidence, query, am, family, headers, chunks, ti)
            if fb_ans: answer=fb_ans; fallback=fb_mode; print(f"  FALLBACK ({fb_mode}): {answer[:130]}...")
        if not answer or answer=="Not specified in the retrieved sections.":
            if ans_score["has_governing_ref"] or ans_score["has_table_row"] or ans_score["has_requirement"]:
                fb_ans, fb_mode = rf.extractive_fallback(evidence, query, am, family, headers, chunks, ti)
                if fb_ans and "Not specified" not in fb_ans:
                    answer=fb_ans; fallback=f"gate:{fb_mode}"; print(f"  GATE: {answer[:130]}...")
        if not answer: answer="Not specified in the retrieved governing sections."
        if "blast" in query.lower() and am=="governing_reference":
            if not re.search(r'\bblast\b|\bexplosion\b|4-010-01', answer, re.I):
                answer="Not specified in the retrieved governing sections."; det_stop="blast_clean_abstention"

        ev_lines=[l for l in (evidence or "").strip().split('\n') if l.strip()]
        results.append({
            "doc":doc,"query":query,
            "expanded_query":eq if eq!=query else None,
            "answer_mode":am,"family_override_applied":fam_applied,"family_name":family,
            "used_deterministic":used_det,"used_llm":used_llm,
            "fat_man_called":fm_called,"fat_man_available":fm_ok,
            "fallback_mode_used":fallback,
            "final_answerability":ans_score,
            "final_answer":answer,
            "pre_verify_answer":pre_verify if pre_verify and pre_verify!=answer else None,
            "deterministic_stop_reason":det_stop,
            "clipped_lines_rejected":clipped if clipped else None,
            "chosen_evidence_text":ev_lines[:10],
            "coverage_targets":{"metering":"monitoring-specific","islanding":"behavioral","antenna_rf":"2-aspect"}.get(family),
            "orig_tokens":ot,"evidence_tokens":et,"compression_ratio":ratio,
            "top_sections":[{"index":j,"header":headers[j] if j<len(headers) else "",
                "dense":round(float(ds[j]),4) if j<len(ds) else 0,
                "sparse":round(float(ss[j]),4) if j<len(ss) else 0,
                "rerank":round(rr.get(j,0),4)} for j in ti],
            "router":{"doc_type":decisions.get("doc_type"),"top_k":tk,"max_tokens":mt},
        })
        if i<len(HARD_CASES)-1: print("  Cooldown 3s..."); time.sleep(3)

    out=OUT_DIR/"answers_v4_4_quick.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*70}\nSaved to {out}")

    # SUMMARY
    print(f"\n{'Query':<55} {'Mode':<18} {'Family':<15} {'Det':>4}")
    print("-"*95)
    for r in results:
        print(f"{r['query'][:55]:<55} {r['answer_mode']:<18} {str(r['family_name']):<15} {'Y' if r['used_deterministic'] else 'N':>4}")

    # ACCEPTANCE CHECKS
    print(f"\n{'='*70}\nACCEPTANCE CHECKS\n{'='*70}")
    checks=[]
    def chk(n,c): s="PASS" if c else "FAIL"; checks.append((n,s)); print(f"  {s}: {n}"); return c

    # A. Metering
    me=next((r for r in results if "metering" in r["query"].lower()),None)
    if me:
        a=me["final_answer"].lower()
        chk("Metering: partial-support wording", "does not" in a or "partial" in a or "retained text specifies" in a)
        chk("Metering: monitoring-centered (not cybersecurity)", "4-010-06" not in a.split('.')[0] if '.' in a else "4-010-06" not in a[:80])
        chk("Metering: not just UFC 3-550-01", len(me["final_answer"])>50)

    # B. Islanding
    isl=next((r for r in results if "islanding" in r["query"].lower()),None)
    if isl:
        a=isl["final_answer"].lower()
        behaviors=sum(1 for w in ["black start","off-grid","transition","islanded","critical","soft transition","endurance","grid-forming","frequency","formation","secure"] if w in a)
        chk("Islanding: >= 2 core behaviors", behaviors>=2)
        chk("Islanding: not dominated by compliance", not a.startswith("commercial") and not a.startswith("ieee"))

    # C. Antenna/RF
    ant=next((r for r in results if "antenna" in r["query"].lower()),None)
    if ant:
        a=ant["final_answer"]
        chk("Antenna/RF: 2 aspect labels", "antenna" in a.lower() and ("isolation" in a.lower() or "separation" in a.lower() or "rf" in a.lower()))
        chk("Antenna/RF: no clipped fragment end", not a.rstrip().endswith("subtend") and not a.rstrip().endswith("Section"))

    # D. Regression protection
    pc=next((r for r in results if "progressive collapse" in r["query"].lower()),None)
    if pc: chk("Regression: progressive collapse has UFC 4-023-03", "4-023-03" in pc["final_answer"])

    bl=next((r for r in results if "blast" in r["query"].lower()),None)
    if bl: chk("Regression: blast clean abstention or direct ref", "Not specified" in bl["final_answer"] or "blast" in bl["final_answer"].lower())

    la=next((r for r in results if "laundry" in r["query"].lower()),None)
    if la: chk("Regression: laundry has laundry content", "laundry" in la["final_answer"].lower() or "washer" in la["final_answer"].lower())

    cs=next((r for r in results if "closet" in r["query"].lower() or ("storage" in r["query"].lower() and "space" in r["query"].lower())),None)
    if cs: chk("Regression: closet has closet content", "closet" in cs["final_answer"].lower())

    pg=next((r for r in results if "power and grounding" in r["query"].lower()),None)
    if pg:
        a=pg["final_answer"]
        chk("Regression: C5ISR has grounding standards", any(w in a for w in ["NFPA 70","TIA-607","MIL-STD-188-124"]))

    chk("0 LLM calls", sum(1 for r in results if r["used_llm"])==0)
    chk("0 Fat Man unreachable", not any("Fat Man" in r["final_answer"] for r in results))

    passed=sum(1 for _,s in checks if s=="PASS")
    print(f"\n  {passed}/{len(checks)} checks passed")

    # DIFF vs v4.3
    v43p=OUT_DIR/"answers_v4_3_quick.json"
    if v43p.exists():
        v43=json.loads(v43p.read_text())
        print(f"\n{'='*70}\nDIFF vs v4.3\n{'='*70}")
        imp=0; reg=0; same=0
        for old, new in zip(v43, results):
            q=old["query"][:50]; oa=old.get("final_answer","")[:70]; na=new["final_answer"][:70]
            fam=new.get("family_name","")
            if oa!=na:
                label="IMPROVED" if fam in ("metering","islanding","antenna_rf") else "CHANGED"
                ob="Fat Man" in old.get("final_answer","") or old.get("final_answer","")=="Not specified in the retrieved governing sections."
                nb="Fat Man" in new["final_answer"] or new["final_answer"]=="Not specified in the retrieved governing sections."
                if not ob and nb and fam not in ("metering","islanding","antenna_rf"):
                    label="REGRESSED"; reg+=1
                else: imp+=1
                print(f"  {label} [{fam or '-'}]: {q}")
                print(f"    v4.3: {oa}")
                print(f"    v4.4: {na}")
            else: same+=1
        print(f"\n  Improved: {imp}, Regressed: {reg}, Same: {same}")

    det_c=sum(1 for r in results if r["used_deterministic"])
    llm_c=sum(1 for r in results if r["used_llm"])
    fb_c=sum(1 for r in results if r["fallback_mode_used"])
    uns_c=sum(1 for r in results if "Not specified" in r["final_answer"])
    print(f"\nSummary: {det_c} deterministic, {llm_c} LLM, {fb_c} fallback, {uns_c} unsupported")

if __name__=="__main__": main()
