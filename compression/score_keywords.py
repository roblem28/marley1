#!/usr/bin/env python3
"""
score_keywords.py — Keyword-based quality scorer for Marley1 eval.
Scores compressed vs baseline answers against known ground-truth keywords.
No LLM required.
"""
import json
from pathlib import Path

RESULTS_IN  = Path("~/marley1/compression/eval_results_v2.json").expanduser()
RESULTS_OUT = Path("~/marley1/compression/eval_results_scored.json").expanduser()

# Ground truth keywords per doc — terms that MUST appear in a good answer
GROUND_TRUTH = {
    "ufc_electrical": [
        "underwriters", "nfpa", "redundan", "generator", "ups",
        "electromagnetic", "mil-std", "circuit", "voltage", "grounding"
    ],
    "ufc_structural": [
        "asce", "seismic", "blast", "reinforced", "concrete",
        "load", "progressive collapse", "ibc", "lateral", "foundation"
    ],
    "ufc_microgrid": [
        "islanded", "grid connected", "frequency", "renewable",
        "energy storage", "diesel", "photovoltaic", "resilience",
        "demand response", "black start"
    ],
    "ufc_c5isr_facilities": [
        "grade 3", "grade 4", "redundan", "concurrently maintainable",
        "fault tolerant", "ups", "generator", "hvac", "emc", "secure"
    ],
    "ufc_unaccompanied_housing": [
        "ndaa", "secretary of defense", "barracks", "single occupancy",
        "ada", "privacy", "energy", "square feet", "bf", "standard"
    ],
}

def score(answer, keywords):
    answer_lower = answer.lower()
    hits = [kw for kw in keywords if kw.lower() in answer_lower]
    return round(len(hits) / len(keywords), 2), hits

def main():
    results = json.loads(RESULTS_IN.read_text())
    scored = []

    print(f"\n{'='*80}")
    print(f"{'Doc':<30} {'Ratio':>6}  {'Comp':>6}  {'Base':>6}  {'Delta':>6}  {'Winner'}")
    print("-"*80)

    for r in results:
        doc = r["doc"]
        kws = GROUND_TRUTH.get(doc, [])
        comp_score, comp_hits = score(r["compressed"]["answer"], kws)
        base_score, base_hits = score(r["baseline"]["answer"],   kws)
        delta = round(comp_score - base_score, 2)
        winner = "COMPRESSED" if comp_score > base_score else ("BASELINE" if base_score > comp_score else "TIE")

        print(f"{doc:<30} {r['compression_ratio']:>5}x  {comp_score:>6.2f}  {base_score:>6.2f}  {delta:>+6.2f}  {winner}")

        scored.append({**r,
            "keyword_scores": {
                "compressed": comp_score, "compressed_hits": comp_hits,
                "baseline":   base_score, "baseline_hits":   base_hits,
                "delta": delta, "winner": winner,
            }
        })

    RESULTS_OUT.write_text(json.dumps(scored, indent=2))
    print(f"\nSaved to {RESULTS_OUT}")

    # Summary
    avg_comp = round(sum(s["keyword_scores"]["compressed"] for s in scored) / len(scored), 2)
    avg_base = round(sum(s["keyword_scores"]["baseline"]   for s in scored) / len(scored), 2)
    avg_ratio = round(sum(s["compression_ratio"] for s in scored) / len(scored), 1)
    wins = sum(1 for s in scored if s["keyword_scores"]["winner"] == "COMPRESSED")
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"  Avg compression ratio : {avg_ratio}x")
    print(f"  Avg keyword score     : compressed={avg_comp}  baseline={avg_base}")
    print(f"  Compressed wins       : {wins}/{len(scored)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
