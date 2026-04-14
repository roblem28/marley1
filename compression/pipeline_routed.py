#!/usr/bin/env python3
"""Marley1 Routed Pipeline.
Router (symbolic) -> Relevance Filter -> Abbreviation -> Fat Man inference.
All pipeline parameters set by symbolic rules before any neural compute.
"""
import json, sys, os, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pdfplumber
from router import route, cache_set
from relevance_filter import RelevanceFilter
from abbrev import ConstructionCompressor

FAT_MAN_URL = "http://100.97.87.86:8080/v1/chat/completions"

def word_count(text):
    return len(text.split())

def extract_pdf(path):
    import hashlib
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "text")
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5((path + str(os.path.getmtime(path))).encode()).hexdigest()
    cache_file = os.path.join(cache_dir, key + ".txt")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return f.read()
    pdf = pdfplumber.open(path)
    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    pdf.close()
    with open(cache_file, "w") as f:
        f.write(text)
    return text

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

def run(pdf_path, query):
    t_start = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"DOC:   {os.path.basename(pdf_path)}")
    print(f"QUERY: {query}")

    t0 = time.perf_counter()
    full_text = extract_pdf(pdf_path)
    t_extract = time.perf_counter() - t0
    orig_tokens = word_count(full_text)
    print(f"  Extracted ~{orig_tokens:,} tokens in {t_extract:.1f}s")

    t0 = time.perf_counter()
    decisions = route(full_text, query, pdf_path)
    t_route = time.perf_counter() - t0
    dt = decisions.get('doc_type', '?')
    cx = decisions.get('query_complexity', '?')
    cs = decisions.get('chunk_size', '?')
    tk = decisions.get('top_k', '?')
    mt = decisions.get('max_tokens', '?')
    print(f"  Router: type={dt} complexity={cx} chunks={cs} topk={tk} max_tokens={mt} ({t_route*1000:.0f}ms)")

    if decisions.get("cache_hit"):
        print("  CACHE HIT")
        return decisions["cached_answer"], {"cached": True}

    if not decisions.get("fatman_healthy", True):
        print("  WARNING: Fat Man not responding")
        return None, {"error": "fatman_down"}

    t0 = time.perf_counter()
    # Try daemon first (fast path)
    try:
        import urllib.request, json as _json
        pdf_name = os.path.basename(pdf_path)
        req_body = _json.dumps({"pdf": pdf_name, "query": query, "top_k": min(decisions["top_k"], 5)}).encode()
        req = urllib.request.Request("http://localhost:8091/filter",
              data=req_body, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=5)
        daemon_result = _json.loads(resp.read())
        filtered_text = daemon_result["context"]
        print(f"  Daemon filter: {daemon_result['elapsed']:.3f}s ({daemon_result['n_chunks']} chunks)")
    except Exception as _e:
        print(f"  Daemon unavailable ({_e}), falling back to local filter")
        rf = RelevanceFilter()
        chunks = rf.chunk(full_text, chunk_size=decisions["chunk_size"])
        kept_indices, similarities = rf.filter(chunks, query, top_k=decisions["top_k"])
        filtered_text = " ".join(chunks[int(i)] for i in kept_indices)
    t_filter = time.perf_counter() - t0
    comp_tokens = word_count(filtered_text)
    ratio = round(orig_tokens / comp_tokens, 1) if comp_tokens else 0
    print(f"  Compressed {orig_tokens:,} -> {comp_tokens:,} tokens ({ratio}x) in {t_filter:.1f}s")

    if not decisions.get("skip_abbrev", False):
        abbrev = ConstructionCompressor()
        filtered_text = abbrev.compress(filtered_text)
        print("  Abbreviation applied")
    else:
        print("  Abbreviation: skipped (router)")

    t0 = time.perf_counter()
    prompt = f"Question: {query}\n\nContext:\n{filtered_text}"
    answer, usage = chat(
        [{"role": "user", "content": prompt}],
        system=decisions["system_prompt"],
        max_tokens=decisions["max_tokens"]
    )
    t_inference = time.perf_counter() - t0
    pt = usage.get('prompt_tokens', '?')
    print(f"  Inference: {t_inference:.1f}s, {pt} prompt tokens")

    t_total = time.perf_counter() - t_start
    metadata = {
        "doc_type": decisions.get("doc_type"),
        "query_complexity": decisions.get("query_complexity"),
        "orig_tokens": orig_tokens,
        "comp_tokens": comp_tokens,
        "ratio": ratio,
        "total_time": t_total,
        "prompt_tokens": usage.get("prompt_tokens"),
    }
    cache_set(pdf_path, query, answer, metadata)
    print(f"  Total: {t_total:.1f}s")
    print(f"\n  ANSWER:")
    print(f"  {'-'*50}")
    for line in answer.split("\n"):
        print(f"  {line}")
    print(f"  {'-'*50}")
    return answer, metadata

if __name__ == "__main__":
    if len(sys.argv) == 3:
        run(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 pipeline_routed.py <pdf> <query>")
