#!/usr/bin/env python3
"""LLMLingua-2 compression test -- compare with abbrev.py on benchmark corpus."""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llmlingua import PromptCompressor
from abbrev import ConstructionCompressor

# Same 5 paragraphs from benchmark.py
CORPUS = {
    "short_spec_clause": (
        "The general contractor shall coordinate all fire protection and "
        "fire suppression submittals with the subcontractor prior to "
        "installation. All heating ventilation and air conditioning "
        "equipment specifications must reference the approved shop drawings "
        "and comply with the quality control inspection requirements "
        "outlined in the construction documents. The architect of record "
        "shall review each submittal within ten business days of receipt."
    ),
    "medium_rfi": (
        "This request for information pertains to the variable air volume "
        "box routing on Level 3 of Building B. The mechanical subcontractor "
        "has identified a conflict between the air handling unit supply "
        "ductwork and the structural steel framing at gridline J-7. The "
        "heating ventilation and air conditioning drawings indicate a 24-inch "
        "main trunk running east-west, but the as-built drawings for the "
        "structural steel show a W12x26 beam at the same elevation. The "
        "general contractor requests clarification from the engineer of "
        "record on whether the ductwork can be rerouted below the beam "
        "without impacting the ceiling height in the finished space. The "
        "construction manager notes that this conflict is on the critical "
        "path method schedule and any delay will affect the baseline "
        "schedule milestone for substantial completion of the mechanical "
        "systems. Please provide a response within five business days to "
        "avoid potential liquidated damages."
    ),
    "long_commissioning": (
        "The commissioning agent shall develop and execute a comprehensive "
        "commissioning plan for all mechanical and electrical systems in "
        "the new data center facility. This includes functional performance "
        "testing of the computer room air conditioning units, computer room "
        "air handler systems, and the building automation system integration "
        "with the direct digital control network. The uninterruptible power "
        "supply and automatic transfer switch shall be tested under full "
        "load conditions, with the generator performing a four-hour load "
        "bank test. The fire alarm control panel shall be verified for "
        "proper integration with the fire suppression and sprinkler "
        "systems, including smoke detector coverage in the hot aisle "
        "containment and cold aisle containment zones. The variable "
        "frequency drive controllers for all air handling unit supply and "
        "return fans must demonstrate stable operation across the full "
        "range of variable air volume demand. Quality assurance "
        "documentation including operation and maintenance manuals, "
        "as-built drawings, and record drawings shall be submitted to the "
        "owner's representative before the certificate of occupancy "
        "inspection. The construction manager and general contractor must "
        "ensure all punch list items related to the building automation "
        "system, energy management system, and switchgear are resolved "
        "prior to final completion."
    ),
    "dc_mep_narrative": (
        "The data center mechanical and electrical plant comprises dual "
        "redundancy N+1 computer room air handler units served by a chilled "
        "water loop with variable frequency drive controlled pumps. Each "
        "power distribution unit feeds six remote power panel boards "
        "supporting the raised access floor server racks. The "
        "uninterruptible power supply system is configured as 2N with "
        "static transfer switch failover capability. The battery monitoring "
        "system provides real-time status to the building management system "
        "via the direct digital control network. Structured cabling and "
        "fiber optic backbone runs are routed through dedicated cable tray "
        "pathways above the hot aisle containment structures. Environmental "
        "monitoring sensors are deployed at every third rack unit for "
        "temperature and humidity tracking. The fire protection strategy "
        "includes a clean agent fire suppression system with smoke detector "
        "coverage at ceiling level and below the raised access floor. The "
        "quality control team has verified all switchgear and transformer "
        "installations against the approved shop drawings and "
        "specifications. The commissioning plan addresses integrated "
        "systems testing of the building automation system, fire alarm "
        "control panel, and the energy management system."
    ),
    "schedule_analysis": (
        "The baseline schedule update for Phase 2 reflects a critical path "
        "method analysis showing 14 days of negative float on the "
        "mechanical rough-in sequence. The work breakdown structure has "
        "been revised to separate the heating ventilation and air "
        "conditioning installation from the plumbing and fire protection "
        "activities, allowing the subcontractor to mobilize additional "
        "crews. The schedule of values indicates 62 percent complete for "
        "the electrical distribution, including the motor control center, "
        "switchgear, and transformer installations. The construction "
        "manager has requested a two-week look-ahead schedule from the "
        "general contractor showing all predecessor and successor "
        "relationships for the data center raised access floor and "
        "structured cabling milestones. The schedule performance index "
        "currently stands at 0.91 and the cost performance index at 0.97. "
        "Earned value analysis suggests the total project cost will exceed "
        "the guaranteed maximum price unless the subcontractor accelerates "
        "the quality control inspections on the remaining submittals and "
        "shop drawings. Liquidated damages will apply if substantial "
        "completion is not achieved by the contractual milestone date."
    ),
}

TARGET_RATIOS = [0.5, 0.2, 0.1]


def count_tokens_approx(text):
    """Rough word-level token count."""
    return len(text.split())


def main():
    print("=" * 70)
    print("  LLMLingua-2 Compression Test")
    print("=" * 70)
    print("\nLoading LLMLingua-2 model (this may take a minute on Pi 5)...")

    t0 = time.perf_counter()
    lingua = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True,
        device_map="cpu",
    )
    load_time = time.perf_counter() - t0
    print("Model loaded in {:.1f}s\n".format(load_time))

    # Also prepare abbrev compressor for comparison
    abbrev = ConstructionCompressor()

    sep = "-" * 70

    for para_name, para_text in CORPUS.items():
        print("=" * 70)
        print("  PARAGRAPH: {}".format(para_name))
        print("  Original length: {} chars, ~{} words".format(
            len(para_text), count_tokens_approx(para_text)))
        print("=" * 70)

        # --- Abbrev.py baseline ---
        t_start = time.perf_counter()
        abbrev_result = abbrev.compress(para_text)
        t_abbrev = time.perf_counter() - t_start
        abbrev_ratio = len(abbrev_result) / len(para_text)
        print("\n  [abbrev.py]  chars: {} -> {}  ratio: {:.3f}  time: {:.1f}ms".format(
            len(para_text), len(abbrev_result), abbrev_ratio, t_abbrev * 1000))
        if len(abbrev_result) > 200:
            print("  Result: {}...".format(abbrev_result[:200]))
        else:
            print("  Result: {}".format(abbrev_result))

        # --- LLMLingua-2 at each target ratio ---
        for ratio in TARGET_RATIOS:
            print("\n  [LLMLingua-2 target_token={}]".format(ratio))
            t_start = time.perf_counter()
            result = lingua.compress_prompt(
                [para_text],
                rate=ratio,
                force_tokens=['\n', '.', ',', '?', '!'],
            )
            t_elapsed = time.perf_counter() - t_start

            compressed = result["compressed_prompt"]
            orig_tokens = result.get("origin_tokens", count_tokens_approx(para_text))
            comp_tokens = result.get("compressed_tokens", count_tokens_approx(compressed))
            actual_ratio = comp_tokens / orig_tokens if orig_tokens else 0

            print("  Original tokens : {}".format(orig_tokens))
            print("  Compressed tokens: {}".format(comp_tokens))
            print("  Actual ratio     : {:.3f}".format(actual_ratio))
            print("  Time             : {:.2f}s".format(t_elapsed))
            if len(compressed) > 300:
                print("  Compressed text  : {}...".format(compressed[:300]))
            else:
                print("  Compressed text  : {}".format(compressed))

        print("\n{}".format(sep))

    print("\nDone.")


if __name__ == "__main__":
    main()
