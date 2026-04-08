#!/usr/bin/env python3
"""Benchmark harness for Qwen2.5-3B on Fat Man.

Tests 6 configurations across 5 construction-domain paragraphs, measuring
throughput and compression metrics via llama.cpp /v1/chat/completions.
TurboQuant (q4_0 KV cache) is now a server-side flag, not per-request.
"""

import argparse
import json
import sys
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abbrev import ConstructionCompressor

# ---------------------------------------------------------------------------
# Test Corpus
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 6 configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    "BASELINE",
    "ABBREV_ONLY",
    "TURBOQUANT_ONLY",
    "COMBINED",
    "ABBREV_NO_DICT",
    "COMBINED_NO_DICT",
]

# Which configs use abbreviation compression
ABBREV_CONFIGS = {"ABBREV_ONLY", "COMBINED", "ABBREV_NO_DICT", "COMBINED_NO_DICT"}

# Which configs include the full dictionary in system prompt
DICT_CONFIGS = {"BASELINE", "ABBREV_ONLY", "TURBOQUANT_ONLY", "COMBINED"}

MINIMAL_SYSTEM_PROMPT = (
    "You are a construction industry assistant. Summarize the construction "
    "update and identify any critical issues. The text may contain standard "
    "construction abbreviations."
)


@dataclass
class RequestResult:
    config: str
    paragraph: str
    iteration: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    ttft_seconds: Optional[float] = None
    total_seconds: Optional[float] = None
    tokens_per_second: Optional[float] = None
    compression_ratio: Optional[float] = None
    original_chars: Optional[int] = None
    compressed_chars: Optional[int] = None
    error: Optional[str] = None


def build_system_prompt(compressor: ConstructionCompressor) -> str:
    """Create a system prompt with the full abbreviation dictionary."""
    lines = [
        "You are a construction industry assistant. The user's message may "
        "contain abbreviated terms. Here is the full abbreviation dictionary — "
        "use it to understand any shortened terms:\n"
    ]
    for term, abbrev in sorted(compressor._terms.items()):
        lines.append(f"  {abbrev} = {term}")
    lines.append(
        "\nWhen you see these abbreviations, interpret them as the full terms. "
        "Respond helpfully and accurately about the construction topic described."
    )
    return "\n".join(lines)


def chat_completion(
    server: str,
    user_content: str,
    system_content: str,
) -> dict:
    """Send a non-streaming request and measure timing and token usage."""
    url = f"{server}/v1/chat/completions"

    body = {
        "model": "qwen2.5-3b",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "max_tokens": 256,
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        t_end = time.perf_counter()

        obj = json.loads(raw)
        usage = obj.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total = t_end - t_start

        tps = completion_tokens / total if completion_tokens and total > 0 else None

        return {
            "total": total,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tps,
            "error": None,
        }

    except Exception as e:
        t_end = time.perf_counter()
        return {
            "total": t_end - t_start,
            "prompt_tokens": None,
            "completion_tokens": None,
            "tokens_per_second": None,
            "error": str(e),
        }


def run_benchmark(server: str, iterations: int) -> list[RequestResult]:
    """Execute all config x paragraph x iteration combinations."""
    compressor = ConstructionCompressor()
    dict_sys_prompt = build_system_prompt(compressor)

    results: list[RequestResult] = []
    total_runs = len(CONFIGS) * len(CORPUS) * iterations
    run_idx = 0

    for config in CONFIGS:
        use_abbrev = config in ABBREV_CONFIGS
        use_dict = config in DICT_CONFIGS
        sys_prompt = dict_sys_prompt if use_dict else MINIMAL_SYSTEM_PROMPT

        for para_name, para_text in CORPUS.items():
            for it in range(1, iterations + 1):
                run_idx += 1
                print(
                    f"  [{run_idx}/{total_runs}] {config:20s} | "
                    f"{para_name:25s} | iter {it}",
                    flush=True,
                )

                if use_abbrev:
                    user_text = compressor.compress(para_text)
                    comp_ratio = len(para_text) / len(user_text) if len(user_text) else None
                    orig_chars = len(para_text)
                    comp_chars = len(user_text)
                else:
                    user_text = para_text
                    comp_ratio = None
                    orig_chars = len(para_text)
                    comp_chars = len(para_text)

                user_content = (
                    "Summarize the following construction update and identify "
                    "any critical issues:\n\n" + user_text
                )

                metrics = chat_completion(
                    server=server,
                    user_content=user_content,
                    system_content=sys_prompt,
                )

                result = RequestResult(
                    config=config,
                    paragraph=para_name,
                    iteration=it,
                    prompt_tokens=metrics.get("prompt_tokens"),
                    completion_tokens=metrics.get("completion_tokens"),
                    total_seconds=metrics.get("total"),
                    tokens_per_second=metrics.get("tokens_per_second"),
                    compression_ratio=comp_ratio,
                    original_chars=orig_chars,
                    compressed_chars=comp_chars,
                    error=metrics.get("error"),
                )
                results.append(result)

    return results


def print_summary_table(results: list[RequestResult], iterations: int):
    """Print an aligned summary table averaging across iterations."""
    from collections import defaultdict

    groups: dict[tuple[str, str], list[RequestResult]] = defaultdict(list)
    for r in results:
        groups[(r.config, r.paragraph)].append(r)

    hdr = (
        f"{'Config':<20s} {'Paragraph':<25s} "
        f"{'PromTok':>8s} {'CompTok':>8s} "
        f"{'Total(s)':>9s} "
        f"{'Tok/s':>7s} {'CmpRatio':>9s}"
    )
    sep = "-" * len(hdr)
    print("\n" + sep)
    print("  BENCHMARK RESULTS (averaged over 3 iterations)")
    print(sep)
    print(hdr)
    print(sep)

    def avg(vals):
        clean = [v for v in vals if v is not None]
        return sum(clean) / len(clean) if clean else None

    def fmt(v, decimals=2):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(int(v))

    config_aggs: dict[str, list[RequestResult]] = defaultdict(list)

    for (config, para), rlist in sorted(groups.items()):
        config_aggs[config].extend(rlist)
        a_pt = avg([r.prompt_tokens for r in rlist])
        a_ct = avg([r.completion_tokens for r in rlist])
        a_total = avg([r.total_seconds for r in rlist])
        a_tps = avg([r.tokens_per_second for r in rlist])
        a_cr = avg([r.compression_ratio for r in rlist])

        print(
            f"{config:<20s} {para:<25s} "
            f"{fmt(a_pt, 0):>8s} {fmt(a_ct, 0):>8s} "
            f"{fmt(a_total):>9s} "
            f"{fmt(a_tps):>7s} {fmt(a_cr):>9s}"
        )

    print(sep)
    print("  CONFIG AVERAGES")
    print(sep)
    for config in CONFIGS:
        rlist = config_aggs.get(config, [])
        if not rlist:
            continue
        a_pt = avg([r.prompt_tokens for r in rlist])
        a_ct = avg([r.completion_tokens for r in rlist])
        a_total = avg([r.total_seconds for r in rlist])
        a_tps = avg([r.tokens_per_second for r in rlist])
        a_cr = avg([r.compression_ratio for r in rlist])

        print(
            f"{config:<20s} {'(all paragraphs)':<25s} "
            f"{fmt(a_pt, 0):>8s} {fmt(a_ct, 0):>8s} "
            f"{fmt(a_total):>9s} "
            f"{fmt(a_tps):>7s} {fmt(a_cr):>9s}"
        )

    print(sep + "\n")


def save_results(results: list[RequestResult], path: str):
    """Write raw results to JSON."""
    data = {
        "benchmark": "qwen2.5-3b-compression-v2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "server_flags": "--cache-type-k q4_0 --cache-type-v q4_0",
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen2.5-3B across compression configurations"
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of iterations per config/paragraph (default: 3)",
    )
    parser.add_argument(
        "--server", type=str, default="http://192.168.88.15:8080",
        help="llama.cpp server URL (default: http://192.168.88.15:8080)",
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.expanduser("~/marley1/compression/benchmark_results.json"),
        help="Path for JSON results file",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Qwen2.5-3B Compression Benchmark v2")
    print(f"  Server     : {args.server}")
    print(f"  Server KV  : --cache-type-k q4_0 --cache-type-v q4_0")
    print(f"  Iterations : {args.iterations}")
    print(f"  Configs    : {', '.join(CONFIGS)}")
    print(f"  Paragraphs : {len(CORPUS)}")
    print(f"  Total runs : {len(CONFIGS) * len(CORPUS) * args.iterations}")
    print("=" * 70 + "\n")

    results = run_benchmark(server=args.server, iterations=args.iterations)
    print_summary_table(results, args.iterations)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
