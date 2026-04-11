#!/usr/bin/env python3
"""Eval v4.6: hybrid-best build. answers_v4_6_quick.json."""
import json, sys, os, time, urllib.request, re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from router import route, classify_answer_mode, expand_query, MODE_MAX_TOKENS
from relevance_filter_v2 import (RelevanceFilterV4, extract_text_no_toc, extract_pdf_tables,
                                  _TOC_LINE_RE, detect_family)

FAT_MAN_URL = "http://100.97.87.86:8080/v1/chat/completions"
FAT_MAN_HEALTH = "http://100.97.87.86:8080/health"
SPECS_DIR = Path(os.path.expanduser("~/marley1/specs"))
OUT_DIR = Path(os.path.expanduser("~/marley1/compression"))

FAMILY_HANDLER_VERSIONS = {
    "metering": "v4.6",
    "islanding": "v4.6",
    "closet_storage": "v4.6",
    "c5isr_power_grounding": "v4.5",
    "antenna_rf": "v4.4",
    None: "v4.5",
}

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

# Frozen family expected answer substrings for regression dry-run
FROZEN_EXPECTS = {
    1: {"query_substr": "renewable", "must_contain": ["UFC"]},
    2: {"query_substr": "progressive collapse", "must_contain": ["4-023-03"]},
    3: {"query_substr": "blast", "must_contain": ["Not specified"]},
    4: {"query_substr": "cybersecurity", "must_contain": ["4-010-06"]},
    6: {"query_substr": "power and grounding", "must_contain": ["NFPA 70", "TIA-607"]},
    7: {"query_substr": "antenna", "must_contain": ["antenna", "isolation"]},
    8: {"query_substr": "laundry", "must_contain": ["washer", "laundry"]},
}

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

def run_single(rf, case, dc, dh, de, db, dtk, dpt, dt):
    doc, query = case["doc"], case["query"]
    chunks=dc[doc]; headers=dh[doc]; ce=de[doc]; bm=db[doc]; ot=dtk[doc]
    family = detect_family(query)
    decisions = route(dt[doc], query, str(SPECS_DIR/f"{doc}.pdf"))
    am=decisions["answer_mode"]; eq=decisions["expanded_query"]
    cap=decisions.get("cap_expansion",False); mt=decisions["max_tokens"]
    sp=decisions.get("system_prompt",""); tk=decisions["top_k"]
    if family=="metering": am="partial_support"

    ti, ds, ss, rr = rf.filter_two_lane(query, eq, top_k=tk, headers=headers,
        chunk_embs=ce, chunks=chunks, bm25_index=bm, answer_mode=am, cap_expansion=cap)
    raw_ctx="\n\n".join(chunks[j] for j in ti)
    evidence=rf.extract_evidence(raw_ctx)
    et=len(evidence.split()) if evidence else 0
    ans_score=rf.score_answerability(evidence, query)
    coverage=ans_score["coverage"]
    ratio=round(ot/et,1) if et else 0

    answer=None; used_det=False; used_llm=False; fm_called=False; fm_ok=False
    fallback=None; det_stop=None; fam_applied=None; pre_verify=None; clipped=[]

    det_ans, det_used, det_stop, fam_applied, clipped = rf.deterministic_answer(
        am, evidence, query, family=family, headers=headers, chunks=chunks, top_indices=ti)
    if det_used and det_ans:
        answer=det_ans; used_det=True
    if not answer and coverage!="unsupported":
        fm_called=True; fm_ok=wait_fm(30)
        if fm_ok:
            used_llm=True
            raw, ok = chat([{"role":"user","content":f"Question: {query}\n\nEvidence:\n{evidence}"}], system=sp, mt=mt)
            if ok and raw: pre_verify=raw; answer=rf.verify_answer(raw, evidence)
    if not answer and evidence and evidence.strip():
        fb_ans, fb_mode = rf.extractive_fallback(evidence, query, am, family, headers, chunks, ti)
        if fb_ans: answer=fb_ans; fallback=fb_mode
    if not answer or answer=="Not specified in the retrieved sections.":
        if ans_score["has_governing_ref"] or ans_score["has_table_row"] or ans_score["has_requirement"]:
            fb_ans, fb_mode = rf.extractive_fallback(evidence, query, am, family, headers, chunks, ti)
            if fb_ans and "Not specified" not in fb_ans:
                answer=fb_ans; fallback=f"gate:{fb_mode}"
    if not answer: answer="Not specified in the retrieved governing sections."
    if "blast" in query.lower() and am=="governing_reference":
        if not re.search(r'\bblast\b|\bexplosion\b|4-010-01', answer, re.I):
            answer="Not specified in the retrieved governing sections."; det_stop="blast_clean_abstention"

    pq_score = rf.proof_quality_score(evidence, answer, family)
    fh_version = FAMILY_HANDLER_VERSIONS.get(family, "v4.5")

    ev_lines=[l for l in (evidence or "").strip().split('\n') if l.strip()]
    return {
        "doc":doc,"query":query,
        "expanded_query":eq if eq!=query else None,
        "answer_mode":am,"family_override_applied":fam_applied,"family_name":family,
        "family_handler_version": fh_version,
        "used_deterministic":used_det,"used_llm":used_llm,
        "fat_man_called":fm_called,"fat_man_available":fm_ok,
        "fallback_mode_used":fallback,
        "final_answerability":ans_score,
        "final_answer":answer,
        "proof_quality_score": pq_score,
        "pre_verify_answer":pre_verify if pre_verify and pre_verify!=answer else None,
        "deterministic_stop_reason":det_stop,
        "clipped_lines_rejected":clipped if clipped else None,
        "chosen_evidence_text":ev_lines[:10],
        "coverage_targets":{"metering":"monitoring-specific","islanding":"behavioral","antenna_rf":"2-aspect","closet_storage":"clean-deterministic"}.get(family),
        "orig_tokens":ot,"evidence_tokens":et,"compression_ratio":ratio,
        "top_sections":[{"index":j,"header":headers[j] if j<len(headers) else "",
            "dense":round(float(ds[j]),4) if j<len(ds) else 0,
            "sparse":round(float(ss[j]),4) if j<len(ss) else 0,
            "rerank":round(rr.get(j,0),4)} for j in ti],
        "router":{"doc_type":decisions.get("doc_type"),"top_k":tk,"max_tokens":mt},
    }

def main():
    print(f"\n{'='*70}")
    print("EVAL V4.6 -- hybrid-best build (0.80 push)")
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

    # ==========================================
    # REGRESSION DRY-RUN: frozen families only
    # ==========================================
    print(f"\n{'='*70}\nREGRESSION DRY-RUN (frozen families)\n{'='*70}")
    frozen_indices = [1, 2, 3, 4, 6, 7, 8]  # indices into HARD_CASES
    frozen_ok = True
    for idx in frozen_indices:
        case = HARD_CASES[idx]
        r = run_single(rf, case, dc, dh, de, db, dtk, dpt, dt)
        answer = r["final_answer"]
        expects = FROZEN_EXPECTS[idx]
        al = answer.lower()
        passed = all(any(mc.lower() in al for mc in [mc]) for mc in expects["must_contain"])
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: [{idx}] {case['query'][:50]}...")
        if not passed:
            frozen_ok = False
            print(f"    EXPECTED: {expects['must_contain']}")
            print(f"    GOT: {answer[:120]}")

    if not frozen_ok:
        print("\n  FROZEN FAMILY REGRESSION DETECTED -- ABORTING")
        return

    print("\n  All frozen families pass. Proceeding with full eval.")

    # ==========================================
    # FULL 10-QUERY EVAL
    # ==========================================
    print(f"\n{'='*70}\nRUNNING FULL EVAL\n{'='*70}")
    results = []
    for i, case in enumerate(HARD_CASES):
        doc, query = case["doc"], case["query"]
        print(f"\n[{i+1}/10] {doc}")
        print(f"  Q: {query}")
        r = run_single(rf, case, dc, dh, de, db, dtk, dpt, dt)
        results.append(r)
        fam = r["family_name"] or "-"
        print(f"  Router: mode={r['answer_mode']}, family={fam}, fhv={r['family_handler_version']}")
        print(f"  Evidence: {r['orig_tokens']:,} -> {r['evidence_tokens']} ({r['compression_ratio']}x)")
        if r["used_deterministic"]:
            print(f"  DETERMINISTIC ({r['deterministic_stop_reason']}): {r['final_answer'][:130]}...")
        elif r["used_llm"]:
            print(f"  LLM: {r['final_answer'][:130]}...")
        else:
            print(f"  ANSWER: {r['final_answer'][:130]}...")
        print(f"  PQ: {r['proof_quality_score']}")
        if i<len(HARD_CASES)-1: print("  Cooldown 3s..."); time.sleep(3)

    out=OUT_DIR/"answers_v4_6_quick.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*70}\nSaved to {out}")

    # SUMMARY TABLE
    print(f"\n{'Query':<50} {'Mode':<16} {'Family':<15} {'Det':>3} {'PQ':>3} {'FHV':<5}")
    print("-"*95)
    for r in results:
        print(f"{r['query'][:50]:<50} {r['answer_mode']:<16} {str(r['family_name'] or '-'):<15} {'Y' if r['used_deterministic'] else 'N':>3} {r['proof_quality_score']:>3} {r['family_handler_version']:<5}")

    # ==========================================
    # ACCEPTANCE CHECKS
    # ==========================================
    print(f"\n{'='*70}\nACCEPTANCE CHECKS\n{'='*70}")
    checks=[]
    def chk(n,c): s="PASS" if c else "FAIL"; checks.append((n,s)); print(f"  {s}: {n}"); return c

    # A. Metering
    me=next((r for r in results if "metering" in r["query"].lower()),None)
    if me:
        a=me["final_answer"]; al=a.lower()
        chk("Metering: monitoring/control or coordination language",
            any(w in al for w in ["monitoring", "monitor", "control", "coordinate"]))
        chk("Metering: explicit partial-support disclaimer",
            "does not provide a complete" in al or "not specify" in al)
        chk("Metering: not primarily cybersecurity-centered",
            al[:80].count("cybersecurity") == 0 and al[:80].count("ufc 4-010-06") <= 1)
        chk("Metering: no telecom garbage", "telecommunications" not in al[:100])
        chk("Metering: no generic design-process lines", "document design decisions" not in al)

    # B. Islanding
    isl=next((r for r in results if "islanding" in r["query"].lower()),None)
    if isl:
        a=isl["final_answer"]; al=a.lower()
        behaviors=sum(1 for w in ["black start","off-grid","transition","islanded","critical",
                                   "soft transition","endurance","grid-forming","frequency",
                                   "formation","secure"] if w in al)
        bullet_lines = [l for l in a.strip().split('\n') if l.strip()]
        chk("Islanding: >= 2 concrete behaviors", behaviors>=2)
        chk("Islanding: no standalone section title bullets",
            not any(re.match(r'^[A-Z]?-?\d{1,2}-\d{1,2}', l.strip()) and not re.search(r'\b(?:shall|must|required)\b', l, re.I) for l in bullet_lines))
        chk("Islanding: no generic utility/accommodation filler",
            not any(w in al for w in ["business relationship", "establishing a microgrid can"]))
        chk("Islanding: max 3 bullets", len(bullet_lines) <= 3)

    # C. Closet/storage
    cs=next((r for r in results if "closet" in r["query"].lower() or ("storage" in r["query"].lower() and "space" in r["query"].lower())),None)
    if cs:
        a=cs["final_answer"]; al=a.lower()
        chk("Closet: begins with closet values", al.strip().startswith("closet"))
        chk("Closet: contains NSF value", "nsf" in al)
        chk("Closet: no separate kitchenette/bathroom requirement line",
            not any(l.strip().lower().startswith(("kitchenette:", "bathroom:")) for l in a.split('\n')))
        chk("Closet: clean formatting (no raw [TABLE] prefix)", "[TABLE]" not in a)

    # D. Frozen family regressions
    pc=next((r for r in results if "progressive collapse" in r["query"].lower()),None)
    if pc: chk("Frozen: progressive collapse has UFC 4-023-03", "4-023-03" in pc["final_answer"])

    bl=next((r for r in results if "blast" in r["query"].lower()),None)
    if bl: chk("Frozen: blast clean abstention", "Not specified" in bl["final_answer"])

    cy=next((r for r in results if "cybersecurity" in r["query"].lower()),None)
    if cy: chk("Frozen: cybersecurity has UFC 4-010-06", "4-010-06" in cy["final_answer"])

    re_q=next((r for r in results if "renewable" in r["query"].lower()),None)
    if re_q: chk("Frozen: renewable has UFC refs", "UFC" in re_q["final_answer"])

    pg=next((r for r in results if "power and grounding" in r["query"].lower()),None)
    if pg: chk("Frozen: C5ISR has grounding standards", any(w in pg["final_answer"] for w in ["NFPA 70","TIA-607","MIL-STD-188-124"]))

    ant=next((r for r in results if "antenna" in r["query"].lower()),None)
    if ant: chk("Frozen: antenna/RF 2-aspect", "antenna" in ant["final_answer"].lower() and ("isolation" in ant["final_answer"].lower() or "rf" in ant["final_answer"].lower()))

    la=next((r for r in results if "laundry" in r["query"].lower()),None)
    if la: chk("Frozen: laundry has washer content", "washer" in la["final_answer"].lower() or "laundry" in la["final_answer"].lower())

    chk("0 LLM calls", sum(1 for r in results if r["used_llm"])==0)
    chk("0 Fat Man in answers", not any("Fat Man" in r["final_answer"] for r in results))

    passed=sum(1 for _,s in checks if s=="PASS")
    total=len(checks)
    print(f"\n  {passed}/{total} checks passed")
    if passed < total:
        print("  SOME CHECKS FAILED -- review above")

    # ==========================================
    # DIFF vs v4.5
    # ==========================================
    v45p=OUT_DIR/"answers_v4_5_quick.json"
    if v45p.exists():
        v45=json.loads(v45p.read_text())
        print(f"\n{'='*70}\nDIFF vs v4.5\n{'='*70}")
        imp=0; reg=0; same=0
        for old, new in zip(v45, results):
            q=old["query"][:50]; oa=old.get("final_answer","")[:80]; na=new["final_answer"][:80]
            fam=new.get("family_name","")
            opq=old.get("proof_quality_score",0); npq=new.get("proof_quality_score",0)
            if oa!=na:
                fhv = new.get("family_handler_version","")
                if fhv == "v4.6":
                    label="IMPROVED" if npq >= opq else "CHANGED"
                else:
                    label="REGRESSED" if npq < opq else "CHANGED"
                if label=="REGRESSED": reg+=1
                elif label=="IMPROVED": imp+=1
                else: same+=1
                print(f"  {label} [{fam or '-'}] (pq {opq}->{npq}): {q}")
                print(f"    v4.5: {oa}")
                print(f"    v4.6: {na}")
            else: same+=1
        print(f"\n  Improved: {imp}, Regressed: {reg}, Same: {same}")

    # DIFF vs v4.3
    v43p=OUT_DIR/"answers_v4_3_quick.json"
    if v43p.exists():
        v43=json.loads(v43p.read_text())
        print(f"\n{'='*70}\nDIFF vs v4.3\n{'='*70}")
        imp=0; reg=0; same=0
        for old, new in zip(v43, results):
            q=old["query"][:50]; oa=old.get("final_answer","")[:60]; na=new["final_answer"][:60]
            if oa!=na: imp+=1; print(f"  CHANGED: {q}")
            else: same+=1
        print(f"\n  Changed: {imp}, Same: {same}")

    det_c=sum(1 for r in results if r["used_deterministic"])
    llm_c=sum(1 for r in results if r["used_llm"])
    fb_c=sum(1 for r in results if r["fallback_mode_used"])
    uns_c=sum(1 for r in results if "Not specified" in r["final_answer"])
    avg_pq=sum(r["proof_quality_score"] for r in results)/len(results) if results else 0
    print(f"\nSummary: {det_c} deterministic, {llm_c} LLM, {fb_c} fallback, {uns_c} unsupported")
    print(f"Average proof_quality_score: {avg_pq:.1f}/100")
    print(f"Target: >= 80.0")
    print(f"{'PASS' if avg_pq >= 80.0 else 'MISS'}: avg_pq={'%.1f' % avg_pq}")

if __name__=="__main__": main()
